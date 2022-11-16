
import torch
from torch import nn
import torch.nn.functional as F


torch.manual_seed(1)
torch.use_deterministic_algorithms(mode=True)


class CNN(nn.Module):
    def __init__(self, embedding_matrix, n_classes, n_words, embed_size):
        super(CNN, self).__init__()
        filter_sizes = [1,2,3,5]
        num_filters = 20
        self.embedding = nn.Embedding(n_words, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        # Hidden layer is 2/3 the size of input layer
        input_size = len(filter_sizes) * num_filters
        self.fc1 = nn.Linear(input_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [torch.sigmoid(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x
