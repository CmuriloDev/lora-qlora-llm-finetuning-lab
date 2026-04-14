from src.generate_dataset import generate_dataset, save_jsonl
from src.train import train


def main():

    print("Generating dataset...")
    data = generate_dataset(50)
    save_jsonl(data)

    print("Starting training...")
    train()


if __name__ == "__main__":
    main()