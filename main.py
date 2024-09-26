from torch_geometric.datasets import WebQSPDataset

if __name__ == "__main__":

    train_dataset = WebQSPDataset('data/WebQSP', split='train')
