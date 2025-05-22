from sklearn.metrics import accuracy_score
import time
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_lines_txt_np(lines_txt_path):
    """
    Parse a .lines.txt file into (N, 2) lane line arrays.

    Parameters:
        lines_txt_path (str): Path to the annotation file.

    Returns:
        List[np.ndarray]: List of (x, y) lane lines.
    """

    lane_lines = []
    with open(lines_txt_path, 'r') as f:
        for line in f:
            points = [float(x) for x in line.strip().split()]
            coords = np.array([
                [int(points[i]), int(points[i + 1])]
                for i in range(0, len(points), 2)
            ], dtype=np.int32)
            lane_lines.append(coords)
    return lane_lines


def get_lines_endPoints(lane_lines_np):
    """
    Extracts the start and end points [x1, y1, x2, y2] of each lane line.

    Parameters:
        lane_lines_np (List[np.ndarray]): Lane lines as (N, 2) arrays.

    Returns:
        List[List[int]]: Start and end points of each line.
    """

    endPoints = []
    for line in lane_lines_np:
        if line.shape[0] >= 2:
            x1, y1 = line[0]
            x2, y2 = line[-1]
            endPoints.append([x1, y1, x2, y2])
    return endPoints


class CULaneLineTxtDataset(Dataset):
    """
    PyTorch Dataset for loading CULane images and their .lines.txt annotations.

    Parameters:
        base_dir (str): Root folder with images and annotation files.
        image_list_path (str): File listing relative image paths.

    Each sample returns:
        - image tensor: [3, H, W]
        - endpoint tensor: [N, 4] as [x1, y1, x2, y2]
    """

    def __init__(self, base_dir, image_list_path):
        self.base_dir = base_dir
        with open(image_list_path, 'r') as f:
            self.image_rel_paths = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.image_rel_paths)

    def __getitem__(self, idx):
        rel_path = self.image_rel_paths[idx].lstrip(
            '/')
        img_path = os.path.join(self.base_dir, rel_path)

        lines_path = img_path.replace(".jpg", ".lines.txt")

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        lane_lines = parse_lines_txt_np(lines_path)
        endPoints = get_lines_endPoints(lane_lines)

        image_tensor = torch.from_numpy(image).permute(
            2, 0, 1).float() / 255.0
        if len(endPoints) > 0:
            endPoints_tensor = torch.tensor(endPoints, dtype=torch.float32)
        else:
            endPoints_tensor = torch.empty((0, 4), dtype=torch.float32)
        return image_tensor, endPoints_tensor


def association_lines(previous_lines, current_lines,
                      angle_thresh=0.88,
                      center_dist_thresh=70,
                      endpoint_dist_thresh=60,
                      length_ratio_thresh=0.5):
    """
    Match lines between two frames based on direction, center, and length similarity.

    Parameters:
        previous_lines (List[List[float]]): Lines from the previous frame.
        current_lines (List[List[float]]): Lines from the current frame.
        angle_thresh (float): Min direction similarity.
        center_dist_thresh (float): Max center distance.
        endpoint_dist_thresh (float): Max endpoint distance.
        length_ratio_thresh (float): Min length ratio.

    Returns:
        List[Tuple[int, int]]: Matched (prev_idx, curr_idx) pairs.
    """

    def cosine_similarity(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)

    if previous_lines is None or current_lines is None or len(previous_lines) == 0 or len(current_lines) == 0:
        return []

    matches = []
    scored_pairs = []

    for i, l1 in enumerate(previous_lines):

        if isinstance(l1[0], list):
            l1 = [v for pair in l1 for v in pair]

        x1_1, y1_1, x2_1, y2_1 = map(float, l1)

        v1 = [x2_1 - x1_1, y2_1 - y1_1]
        center1 = [(x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2]
        len1 = np.linalg.norm(v1)

        for j, l2 in enumerate(current_lines):
            x1_2, y1_2, x2_2, y2_2 = map(float, l2)

            v2 = [x2_2 - x1_2, y2_2 - y1_2]
            center2 = [(x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2]
            len2 = np.linalg.norm(v2)

            angle_sim = cosine_similarity(v1, v2)
            center_dist = np.linalg.norm(
                np.array(center1) - np.array(center2))
            start_dist = np.linalg.norm([x1_1 - x1_2, y1_1 - y1_2])
            end_dist = np.linalg.norm([x2_1 - x2_2, y2_1 - y2_2])
            length_ratio = min(len1, len2) / max(len1, len2)

            if (angle_sim > angle_thresh and
                center_dist < center_dist_thresh and
                start_dist < endpoint_dist_thresh and
                end_dist < endpoint_dist_thresh and
                    length_ratio > length_ratio_thresh):

                score = angle_sim * (1 - center_dist / 100) * length_ratio
                scored_pairs.append((score, i, j))

    # Greedy matching: best scores first
    scored_pairs.sort(reverse=True)
    matched_prev = set()
    matched_curr = set()

    for score, i, j in scored_pairs:
        if i in matched_prev or j in matched_curr:
            continue
        matches.append((i, j))
        matched_prev.add(i)
        matched_curr.add(j)

    return matches


def extract_features(previous_line, current_line):
    """
    Compute 7 features between two lines.

    Parameters:
        previous_line (List[float]): [x1, y1, x2, y2] of the first line.
        current_line (List[float]): [x1, y1, x2, y2] of the second line.

    Returns:
        List[float]: [cos_sim, center_dist, length_diff, dx1, dy1, dx2, dy2]
    """

    x1_1, y1_1, x2_1, y2_1 = map(float, previous_line)
    x1_2, y1_2, x2_2, y2_2 = map(float, current_line)

    # Direction vectors
    v1 = np.array([x2_1 - x1_1, y2_1 - y1_1])
    v2 = np.array([x2_2 - x1_2, y2_2 - y1_2])

    # Cosine similarity
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)

    # Centers
    c1 = np.array([(x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2])
    c2 = np.array([(x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2])
    center_dist = np.linalg.norm(c1 - c2)

    # Lengths
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    length_diff = abs(len1 - len2)

    # Endpoint differences
    delta_x1 = x1_2 - x1_1
    delta_y1 = y1_2 - y1_1
    delta_x2 = x2_2 - x2_1
    delta_y2 = y2_2 - y2_1

    return [cos_sim, center_dist, length_diff, delta_x1, delta_y1, delta_x2, delta_y2]


def extract_line_features(previous_lines, current_lines, association_model=None):
    """
    Extract features and labels for all line pairs.

    Parameters:
        previous_lines (List[List[float]]): Lines from the previous frame.
        current_lines (List[List[float]]): Lines from the current frame.
        association_model (Optional): Neural net for predicting line matches. If None, uses rule-based matching.

    Returns:
        Tuple[List[List[float]], List[int]]: Feature vectors and binary labels.
    """
    all_features = []
    all_labels = []

    if association_model is None:
        matches = association_lines(previous_lines, current_lines)
    else:
        matches = predict_matches_nn(
            previous_lines, current_lines, association_model)

    for i, match in matches:
        previous_line = previous_lines[i]
        for j, current_line in enumerate(current_lines):
            feature = extract_features(previous_line, current_line)
            label = 1 if match == j else 0
            all_features.append(feature)
            all_labels.append(label)

    return all_features, all_labels


def collect_association_features(lines_Dataloader, association_model=None):
    """
    Collect features and labels from a dataloader for line association training.

    Used to generate training or validation data by extracting geometric features
    between lines across consecutive frames, with labels indicating matches.

    Parameters:
        lines_Dataloader: Iterable yielding (image_batch, line_batch) for each frame.
        association_model (optional): Neural model for line matching. If None, uses rule-based matching.

    Returns:
        Tuple:
            - features (List[List[float]]): Feature vectors for line pairs.
            - labels (List[int]): Binary match labels (1 = match, 0 = non-match).
    """

    all_features = []
    all_labels = []
    previous_lines = None

    for image_batch, lines_batch in lines_Dataloader:
        batch_size = image_batch.shape[0]

        for i in range(batch_size):
            image_tensor = image_batch[i]  # Shape: [3, H, W]
            lines_tensor = lines_batch[i]  # Tensor of shape [N, 4] or empty

            current_lines = lines_tensor.tolist()

            if lines_tensor is None or lines_tensor.nelement() == 0:
                continue

            if previous_lines is not None:
                features, labels = extract_line_features(
                    previous_lines, current_lines, association_model=association_model)

                all_features.extend(features)
                all_labels.extend(labels)

            previous_lines = current_lines

    print("Collected", len(all_features), "training samples.")
    return all_features, all_labels


class LineAssociationModel(nn.Module):
    """
    A simple feedforward neural network for line association.

    Takes a 7-dimensional feature vector describing two lines and
    outputs a probability indicating whether they match.

    Parameters:
        input_dim (int): Dimensionality of input features (default: 7).
        hidden_dim (int): Size of hidden layers (default: 64).

    Forward:
        x (torch.Tensor): Input tensor of shape [B, input_dim].

    Returns:
        torch.Tensor: Output probabilities of shape [B], one per input pair.
    """

    def __init__(self, input_dim=7, hidden_dim=64):
        super(LineAssociationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),          # Output single logit
            nn.Sigmoid()                       # Convert to probability between 0 and 1
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


def validate_model(model, val_loader, criterion):
    """
    Evaluate a model on validation data.

    Performs forward passes on the validation set, computes loss and accuracy,
    and returns the average loss and classification accuracy.

    Parameters:
        model (nn.Module): The trained model to evaluate.
        val_loader (DataLoader): DataLoader yielding (features, labels) for validation.
        criterion: Loss function (e.g., nn.BCELoss).

    Returns:
        Tuple[float, float]: Average loss and accuracy on the validation set.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for features, labels in val_loader:
            if isinstance(features, list):
                features = torch.tensor(features, dtype=torch.float32)
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.float32)

            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    return avg_loss, accuracy


def train_association_model(association_train_loader, val_loader, num_epochs=10, patience=5):
    """
    Train the line association model with early stopping.

    Uses binary cross-entropy loss and Adam optimizer to train the model
    on line-pair features. Tracks training/validation loss and validation
    accuracy per epoch, and applies early stopping based on validation loss.

    Parameters:
    association_train_loader (DataLoader): Training data.
    val_loader (DataLoader): Validation data.
    num_epochs (int): Max training epochs.
    patience (int): Stop early if no improvement.

    Returns:
        Trained LineAssociationModel.
    """
    association_model = LineAssociationModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(association_model.parameters(), lr=1e-3)

    best_model_wts = copy.deepcopy(association_model.state_dict())
    best_val_loss = float('inf')
    no_improve_epochs = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        association_model.train()
        running_loss = 0

        for features, labels in tqdm(association_train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = association_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(association_train_loader)
        val_loss, val_acc = validate_model(
            association_model, val_loader, criterion)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(association_model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best weights
    association_model.load_state_dict(best_model_wts)

    return association_model


def predict_matches_nn(previous_lines, current_lines, association_model, threshold=0.5):
    """
    Predict line matches using a trained neural network model.

    For each line in the previous frame, finds the best-matching line in the
    current frame based on predicted match probability, using a threshold.

    Parameters:
        previous_lines (List[List[float]]): Lines from the previous frame.
        current_lines (List[List[float]]): Lines from the current frame.
        association_model (nn.Module): Trained model for predicting match probability.
        threshold (float): Minimum probability required to consider a match (default: 0.5).

    Returns:
        List[Tuple[int, int]]: List of matched line index pairs (prev_idx, curr_idx).
    """

    association_model.eval()
    matches = []
    used_indices = set()

    for i, line1 in enumerate(previous_lines):
        best_score = -1
        best_match = -1

        for j, line2 in enumerate(current_lines):
            if j in used_indices:
                continue
            # Extract feature vector
            features = extract_features(line1, line2)
            features_tensor = torch.tensor(
                features, dtype=torch.float32).unsqueeze(0).to(device)

            # Predict match probability
            with torch.no_grad():
                score = association_model(features_tensor).item()

            if score > best_score and score > threshold:
                best_score = score
                best_match = j

        # Match line i in previous_lines to best_match in current_lines
        matches.append((i, best_match))
        if best_match != -1:
            used_indices.add(best_match)

    return matches


def draw_matched_lines(previous_image, previous_lines, current_image, current_lines, matches, matched_lines_details_previous, match_id_dict, frame_idx):
    """
    Draw and label matched lane lines between consecutive frames.

    For each matched line pair, assigns or reuses an ID, draws the lines on images
    with color-coded IDs, and displays the result. Handles both first-frame 
    initialization and cross-frame visualization.

    Parameters:
        previous_image (np.ndarray): Image from the previous frame.
        previous_lines (List[List[int]]): Lane lines from the previous frame.
        current_image (np.ndarray): Image from the current frame.
        current_lines (List[List[int]]): Lane lines from the current frame.
        matches (List[Tuple[int, int]]): Matched index pairs (prev_idx, curr_idx).
        matched_lines_details_previous (List[List]): Line-ID info from previous frame (in-place updated).
        match_id_dict (dict): Mapping of (prev_idx, curr_idx) to persistent match ID.
        frame_idx (int): Index of the current frame, used for display.

    Returns:
        None. (Displays annotated frames using matplotlib or OpenCV.)
    """

    def id_to_color(idx):
        np.random.seed(idx)
        return tuple(int(x) for x in np.random.randint(0, 255, 3))

    if previous_lines is not None:
        img1 = previous_image.copy()
        img2 = current_image.copy()
        matched_lines_details_current = []

        used_ids = {item[2] for item in matched_lines_details_previous}
        max_existing_id = max(used_ids) if used_ids else -1

        for i, j in matches:
            if j == -1:
                continue

            try:
                x1a, y1a, x2a, y2a = map(int, previous_lines[i])
                x1b, y1b, x2b, y2b = map(int, current_lines[j])

            except:
                continue

            color, match_id = None, None
            for line in matched_lines_details_previous:
                if line[0] == i:
                    color, match_id = line[1], line[2]
                    break

            if match_id is None:
                match_id = max_existing_id + 1
                max_existing_id += 1
                color = id_to_color(match_id)
            match_id_dict[(i, j)] = match_id

            txa = int((x1a + x2a) / 2)
            tya = int((y1a + y2a) / 2)
            txb = int((x1b + x2b) / 2)
            tyb = int((y1b + y2b) / 2)

            cv2.line(img1, (x1a, y1a), (x2a, y2a), color, 6)
            cv2.putText(img1, str(match_id), (txa, tya - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, color, 7)
            cv2.line(img2, (x1b, y1b), (x2b, y2b), color, 6)
            cv2.putText(img2, str(match_id), (txb, tyb - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, color, 7)

            matched_lines_details_current.append([j, color, match_id])

        matched_lines_details_previous.clear()
        matched_lines_details_previous.extend(matched_lines_details_current)
        show_matched_frames(img1, img2, frame_idx)

    else:
        img = current_image.copy()
        matched_lines_details_previous.clear()

        for idx, line in enumerate(current_lines):
            try:
                x1, y1, x2, y2 = map(int, line)
            except:
                continue
            color = id_to_color(idx)
            match_id = idx
            tx = int((x1 + x2) / 2)
            ty = int((y1 + y2) / 2)
            cv2.line(img, (x1, y1), (x2, y2), color, 6)
            cv2.putText(img, str(match_id), (tx, ty - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, color, 7)
            matched_lines_details_previous.append([idx, color, match_id])

        plt.figure(figsize=(12, 5))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Frame t (initial)")
        plt.axis("off")
        plt.figtext(
            0.5, 0.01, f"Frame t =  {frame_idx}", ha="center", fontsize=12)

        plt.show()


def show_matched_frames(img1, img2, frame_idx):
    """
    Show two frames with matched lines.

    Parameters:
        img1 (np.ndarray): Previous frame (t) with annotations.
        img2 (np.ndarray): Current frame (t+1) with annotations.
        frame_idx (int): Frame index for display.

    Returns:
        None. Displays the frames using matplotlib.
    """

    plt.figure(figsize=(12, 7))
    plt.suptitle(
        "Line Matching Between Consecutive Frames (t and t+1)", fontsize=16)

    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Frame t (previous)")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("Frame t+1 (current)")
    plt.axis("off")

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.figtext(
        0.5, 0.02, f"Frame t =  {frame_idx}", ha="center", fontsize=12)
    plt.show()


def collate_fn_variable_lines(batch):
    """
     Custom collate function for variable-length line data.

     Parameters:
         batch (List[Tuple[Tensor, Tensor]]): List of (image_tensor, lines_tensor) pairs.

     Returns:
         Tuple:
             - images_tensor (Tensor): Batched image tensor of shape [B, 3, H, W].
             - lines_list (List[Tensor]): List of (N, 4) tensors with lane lines per sample.
     """

    images = [item[0] for item in batch]         # list of image tensors
    lines_list = [item[1] for item in batch]     # list of (N, 4) tensors

    # Stack images into a single tensor batch
    images_tensor = torch.stack(images, dim=0)

    return images_tensor, lines_list


def visualize_prediction(lines_Dataloader, association_model=None):
    """
    Visualize matched lane lines across consecutive frames.

    Parameters:
        lines_Dataloader: A PyTorch DataLoader yielding (image_tensor, lines_tensor) pairs.
        association_model (optional): Neural network for matching lines. If None, rule-based matching is used.

    Returns:
        None. Displays matched lines with color-coded IDs and prints match results to console.
    """

    previous_lines = None
    matched_lines_details_previous = []
    frame_idx = 0
    with torch.no_grad():
        for image_batch, lines_batch in lines_Dataloader:

            image_tensor = image_batch[0]           # (3, H, W)
            lines_tensor = lines_batch[0]           # (N, 4)

            # Convert image tensor back to NumPy for OpenCV/visualization
            image_np = (image_tensor.permute(
                1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr_current = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Convert lines to list of [x1, y1, x2, y2]
            current_lines = lines_tensor.cpu().tolist()

            if current_lines is None:
                frame_idx += 1
                continue

            matches = None
            if previous_lines is not None:
                if association_model is None:
                    matches = association_lines(
                        previous_lines, current_lines)
                else:
                    matches = predict_matches_nn(
                        previous_lines, current_lines, association_model)
                match_id_dict = {}

                draw_matched_lines(
                    img_bgr_previous, previous_lines, img_bgr_current, current_lines, matches, matched_lines_details_previous, match_id_dict, frame_idx)

            else:
                draw_matched_lines(
                    None, previous_lines, img_bgr_current, current_lines, matches, matched_lines_details_previous, {}, frame_idx)

            previous_lines = current_lines
            img_bgr_previous = img_bgr_current
            frame_idx += 1


def evaluate_association_methods(lines_Dataloader, association_model, num_batches=10):
    """
    Compare classic rule-based and neural network-based line association methods.

    Measures runtime and matching how often the neural network predicted the same match as the rule-based method across a number of image pairs.

    Args:
        lines_Dataloader: PyTorch DataLoader yielding (image, lines) pairs.
        association_model: Trained neural model for matching.
        num_batches (int): Number of image pairs to evaluate (default: 10).

    Returns:
        None. Prints:
            - Agreement rate (% of identical matches)
            - Average runtime per frame for both methods
    """

    previous_lines = None
    classic_total_time = 0.0
    nn_total_time = 0.0
    nn_matches_all = []
    classic_labels_all = []

    for batch_idx, (image, lines) in enumerate(lines_Dataloader):
        if batch_idx >= num_batches:
            break

        if isinstance(lines[0], torch.Tensor):
            current_lines = [line.tolist() for line in lines[0]]
        else:
            current_lines = lines[0]

        if previous_lines is not None and current_lines is not None and len(previous_lines) > 0:

            start = time.time()
            classic_matches = association_lines(previous_lines, current_lines)
            classic_total_time += time.time() - start

            start = time.time()
            nn_matches = predict_matches_nn(
                previous_lines, current_lines, association_model)
            nn_total_time += time.time() - start

            classic_vec = [-1] * len(previous_lines)
            for i, j in classic_matches:
                classic_vec[i] = j

            nn_vec = [-1] * len(previous_lines)
            for i, j in nn_matches:
                nn_vec[i] = j

            classic_labels_all.extend(classic_vec)
            nn_matches_all.extend(nn_vec)

        previous_lines = current_lines

    if classic_labels_all:
        nn_vs_classic_accuracy = accuracy_score(
            classic_labels_all, nn_matches_all)
        print(f"\nChecked {min(num_batches, batch_idx+1)} image pairs.")
        print(
            f"Neural network made the same match as the classic method {nn_vs_classic_accuracy:.2%} of the time.")
        print(
            f"Classic method took about {classic_total_time / (batch_idx+1):.4f} seconds per image.")
        print(
            f"Neural network took about {nn_total_time / (batch_idx+1):.4f} seconds per image.")
    else:
        print("No line matches were found to compare.")


if __name__ == "__main__":

    # Set the following flags to control script behavior:
    # NOTE: Make sure the model is trained before running visualization or evaluation
    train = False          # Train the model and save it

    # Visualize line matching (works with or without trained model)
    visualize = True
    evaluate = False      # Evaluate NN vs rule-based (requires trained model)

    base_dir = r"C:\Users\DELL\Documents\CULaneDataSet"
    model_path = os.path.join(base_dir, "association_model_2025_5_22.pth")

    # Load a CULane dataset split (train/val/test) into a DataLoader
    def get_loader(list_name, batch_size, shuffle, collate_fn=None, workers=4):
        dataset = CULaneLineTxtDataset(
            base_dir=base_dir, image_list_path=os.path.join(base_dir, "list", list_name))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=workers, pin_memory=True)

    val_loader = get_loader("val.txt", batch_size=16,
                            shuffle=False, collate_fn=collate_fn_variable_lines)
    test_loader = get_loader("test.txt", batch_size=1,
                             shuffle=False, collate_fn=collate_fn_variable_lines)
    train_loader = get_loader("train.txt", batch_size=(1 if not train else 16),
                              shuffle=train, collate_fn=(None if not train else collate_fn_variable_lines))

    if train:
        print("Training")
        # Collect features,labels
        train_features, train_labels = collect_association_features(
            train_loader)
        val_features, val_labels = collect_association_features(val_loader)

        # Convert to tensors
        features_train = torch.tensor(train_features, dtype=torch.float32)
        labels_train = torch.tensor(train_labels, dtype=torch.float32)
        features_val = torch.tensor(val_features, dtype=torch.float32)
        labels_val = torch.tensor(val_labels, dtype=torch.float32)

        # Create DataLoaders for training and validation from pre-extracted features and labels
        train_data = DataLoader(torch.utils.data.TensorDataset(
            features_train, labels_train), batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_data = DataLoader(torch.utils.data.TensorDataset(
            features_val, labels_val), batch_size=16, shuffle=False)

        model = train_association_model(
            train_data, val_data, num_epochs=50, patience=7)
        torch.save(model.state_dict(), model_path)
        print("Model saved.")

    elif visualize or evaluate:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Trained model not found. Please run training first.")

        print("Loading model...")
        model = LineAssociationModel(input_dim=7).to(device)
        model.load_state_dict(torch.load(model_path))

        if evaluate:
            print("Evaluating")
            evaluate_association_methods(test_loader, model, num_batches=100)

        if visualize:
            print("Visualizing")

            # visualize_prediction(test_loader, None)# uncommon when want to visualize only classic matches
            visualize_prediction(test_loader, model)
