import torch
import torchvision.transforms as transforms
from PIL import Image


# -----------------------------
# Generate example predictions
# -----------------------------
def print_examples(model, device, dataset):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Ensure same input shape as training
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    model.eval()
    test_images = [
        ("test_examples/dog.jpg", "Dog on a beach by the ocean"),
        ("test_examples/child.jpg", "Child holding red frisbee outdoors"),
        ("test_examples/bus.png", "Bus driving by parked cars"),
        ("test_examples/boat.png", "A small boat in the ocean"),
        ("test_examples/horse.png", "A cowboy riding a horse in the desert"),
    ]

    for idx, (img_path, correct_caption) in enumerate(test_images, 1):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        image_tensor = transform(image).unsqueeze(0).to(device)
        output_tokens = model.caption_image(image_tensor, dataset.vocab)
        output_caption = " ".join(output_tokens)

        print(f"Example {idx} CORRECT: {correct_caption}")
        print(f"Example {idx} OUTPUT : {output_caption}\n")

    model.train()


# -----------------------------
# Save model checkpoint
# -----------------------------
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# -----------------------------
# Load model checkpoint
# -----------------------------
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint.get("step", 0)
    return step
