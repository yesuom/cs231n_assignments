from tensorflow.python.framework.ops import device_v2
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import tensorflow_datasets as tfds
from torchvision import transforms as T
import cv2
from tqdm.auto import tqdm


def get_similarity_no_loop(text_features, image_features):
    """
    Computes the pairwise cosine similarity between text and image feature vectors.

    Args:
        text_features (torch.Tensor): A tensor of shape (N, D).
        image_features (torch.Tensor): A tensor of shape (M, D).

    Returns:
        torch.Tensor: A similarity matrix of shape (N, M), where each entry (i, j)
        is the cosine similarity between text_features[i] and image_features[j].
    """
    similarity = None
    ############################################################################
    # TODO: Compute the cosine similarity. Do NOT use for loops.               #
    ############################################################################
    dot_pro = torch.matmul(text_features, image_features.T)
    length1 = torch.norm(text_features, dim=1).unsqueeze(1)
    length2 = torch.norm(image_features, dim=1).unsqueeze(1)
    similarity = dot_pro / torch.matmul(length1, length2.T)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return similarity


@torch.no_grad()
def clip_zero_shot_classifier(clip_model, clip_preprocess, images,
                              class_texts, device):
    """Performs zero-shot image classification using a CLIP model.

    Args:
        clip_model (torch.nn.Module): The pre-trained CLIP model for encoding
            images and text.
        clip_preprocess (Callable): A preprocessing function to apply to each
            image before encoding.
        images (List[np.ndarray]): A list of input images as NumPy arrays
            (H x W x C) uint8.
        class_texts (List[str]): A list of class label strings for zero-shot
            classification.
        device (torch.device): The device on which computation should be
            performed. Pass text_tokens to this device before passing it to
            clip_model.

    Returns:
        List[str]: Predicted class label for each image, selected from the
            given class_texts.
    """
    
    pred_classes = []

    ############################################################################
    # TODO: Find the class labels for images.                                  #
    ############################################################################
    processed_images = [
        clip_preprocess(Image.fromarray(img)).unsqueeze(0)
        for img in images
    ]
    images_tensor = torch.cat(processed_images, dim=0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(images_tensor)

    text_tokens = clip.tokenize(class_texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
    similarity = get_similarity_no_loop(text_features, image_features)
    pred_classes_id = torch.argmax(similarity, dim=0).tolist()
    pred_classes = [class_texts[i] for i in pred_classes_id]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return pred_classes
  

class CLIPImageRetriever:
    """
    A simple image retrieval system using CLIP.
    """
    
    @torch.no_grad()
    def __init__(self, clip_model, clip_preprocess, images, device):
        """
        Args:
          clip_model (torch.nn.Module): The pre-trained CLIP model.
          clip_preprocess (Callable): Function to preprocess images.
          images (List[np.ndarray]): List of images as NumPy arrays (H x W x C).
          device (torch.device): The device for model execution.
        """
        ############################################################################
        # TODO: Store all necessary object variables to use in retrieve method.    #
        # Note that you should process all images at once here and avoid repeated  #
        # computation for each text query. You may end up NOT using the above      #
        # similarity function for most compute-optimal implementation.#
        ############################################################################
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.images = images
        self.device = device

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def retrieve(self, query: str, k: int = 2):
        """
        Retrieves the indices of the top-k images most similar to the input text.
        You may find torch.Tensor.topk method useful.

        Args:
            query (str): The text query.
            k (int): Return top k images.

        Returns:
            List[int]: Indices of the top-k most similar images.
        """
        top_indices = []
        ############################################################################
        # TODO: Retrieve the indices of top-k images.                              #
        ############################################################################
        text_tokens = clip.tokenize(query).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens) #(1, D)
        processed_images = [
            self.clip_preprocess(Image.fromarray(img)).unsqueeze(0)
            for img in self.images
        ]
        images_tensor = torch.cat(processed_images, dim=0).to(self.device)
        image_features = self.clip_model.encode_image(images_tensor) #(N, D)
        similarity = get_similarity_no_loop(text_features, image_features)
        top_indices = torch.topk(similarity, k, dim=1)[1].squeeze().tolist()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return top_indices

  
class DavisDataset:
    def __init__(self):
        self.davis = tfds.load('davis/480p', split='validation', as_supervised=False)
        self.img_tsfm = T.Compose([
            T.Resize((480, 480)), T.ToTensor(),
            T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        
      
    def get_sample(self, index):
        assert index < len(self.davis)
        ds_iter = iter(tfds.as_numpy(self.davis))
        for i in range(index+1):
            video = next(ds_iter)
        frames, masks = video['video']['frames'], video['video']['segmentations']
        print(f"video {video['metadata']['video_name'].decode()}  {len(frames)} frames")
        return frames, masks
    
    def process_frames(self, frames, dino_model, device):
        res = []
        for f in frames:
            f = self.img_tsfm(Image.fromarray(f))[None].to(device)
            with torch.no_grad():
              tok = dino_model.get_intermediate_layers(f, n=1)[0]
            res.append(tok[0, 1:])

        res = torch.stack(res)
        return res
    
    def process_masks(self, masks, device):
        res = []
        for m in masks:
            m = cv2.resize(m, (60,60), cv2.INTER_NEAREST)
            res.append(torch.from_numpy(m).long().flatten(-2, -1))
        res = torch.stack(res).to(device)
        return res
    
    def mask_frame_overlay(self, processed_mask, frame):
        H, W = frame.shape[:2]
        mask = processed_mask.detach().cpu().numpy()
        mask = mask.reshape((60, 60))
        mask = cv2.resize(
            mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = create_segmentation_overlay(mask, frame.copy())
        return overlay
        


def create_segmentation_overlay(segmentation_mask, image, alpha=0.5):
    """
    Generate a colored segmentation overlay on top of an RGB image.

    Parameters:
        segmentation_mask (np.ndarray): 2D array of shape (H, W), with class indices.
        image (np.ndarray): 3D array of shape (H, W, 3), RGB image.
        alpha (float): Transparency factor for overlay (0 = only image, 1 = only mask).

    Returns:
        np.ndarray: Image with segmentation overlay (shape: (H, W, 3), dtype: uint8).
    """
    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"

    # Generate deterministic colors for each class using a fixed colormap
    def generate_colormap(n):
        np.random.seed(42)  # For determinism
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # Create a color image for the segmentation mask
    seg_color = colormap[segmentation_mask]  # shape: (H, W, 3)

    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

    return overlay


def compute_iou(pred, gt, num_classes):
    """Compute the mean Intersection over Union (IoU)."""
    iou = 0
    for ci in range(num_classes):
        p = pred == ci
        g = gt == ci
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes


def train_val(model, data_loader, train_optimizer, epoch, epochs, device='cpu'):
    is_train = train_optimizer is not None
    model.train() if is_train else model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.3f}'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num))

    return total_loss / total_num, total_correct_1 / total_num, total_correct_5 / total_num


class DINOSegmentation:
    def __init__(self, device, num_classes: int, inp_dim : int = 384):
        """
        Initialize the DINOSegmentation model.

        This defines a simple neural network designed to  classify DINO feature
        vectors into segmentation classes. It includes model initialization,
        optimizer, and loss function setup.

        Args:
            device (torch.device): Device to run the model on (CPU or CUDA).
            num_classes (int): Number of segmentation classes.
            inp_dim (int, optional): Dimensionality of the input DINO features.
        """

        ############################################################################
        # TODO: Define a very lightweight pytorch model, optimizer, and loss       #
        # function to train classify each DINO feature vector into a seg. class.   #
        # It can be a linear layer or two layer neural network.                    #
        ############################################################################
        self.device = device
        # self.model = torch.nn.Linear(inp_dim, num_classes, device=device).to(self.device)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(inp_dim, 256, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes, device=device)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.batch_size = 256
        self.num_epochs = 20
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        pass

    def train(self, X_train, Y_train, num_iters=500):
        """Train the segmentation model using the provided training data.

        Args:
            X_train (torch.Tensor): Input feature vectors of shape (N, D).
            Y_train (torch.Tensor): Ground truth labels of shape (N,).
            num_iters (int, optional): Number of optimization steps.
        """
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.results = {'train_loss': [], 'train_acc@1': [], 'test_loss': [], 'test_acc@1': []}

        # for i in range(num_iters):
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            train_loss, train_acc_1, _ = train_val(self.model, train_loader, self.optimizer, epoch, self.num_epochs, self.device)
            self.results['train_loss'].append(train_loss)
            self.results['train_acc@1'].append(train_acc_1)

            # test_loss, test_acc_1, _ = train_val(self.model, test_loader, None, epoch, self.num_epochs, device)
            # self.results['test_loss'].append(test_loss)
            # self.results['test_acc@1'].append(test_acc_1)
            # if test_acc_1 > best_acc:
            #     best_acc = test_acc_1
        
        # self.results["best_test_acc"] = best_acc

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def inference(self, X_test):
        """Perform inference on the given test DINO feature vectors.

        Args:
            X_test (torch.Tensor): Input feature vectors of shape (N, D).

        Returns:
            torch.Tensor of shape (N,): Predicted class indices.
        """
        pred_classes = None
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        self.model.eval()
        scores = self.model(X_test) #(N,num_classes)
        pred_classes = torch.argmax(scores, dim=1)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return pred_classes