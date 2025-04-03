import torch
import torch.nn as nn
import torch.nn.functional as F

class SalaryPredictor(nn.Module):
    def __init__(self, n_tokens: int , n_cat_features: int , hid_size: int = 64):
        super().__init__()
        
        # Title Encoder
        self.title_embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        
        self.conv_title = nn.Conv1d(10, 100, 3, stride=1, padding=1)
        self.title_fc = nn.Linear(hid_size-2, hid_size)  # Fully connected layer for title
        self.dropout = nn.Dropout(p=0.1)
        self.batch_norm = nn.BatchNorm1d(10, affine=False)

        # Description Encoder
        self.desc_embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.conv_desc = nn.Conv1d(10, 100, 3, stride=1, padding=1)

        self.desc_fc = nn.Linear(hid_size, hid_size)  # Fully connected layer for description
        
        # Categorical Features Encoder
        self.cat_fc = nn.Linear(n_cat_features, hid_size)  # Fully connected layer for categorical features
        
        # Common Network
        self.common_fc1 = nn.Linear(hid_size * 3, hid_size * 2)  # Combine all branches
        self.common_fc2 = nn.Linear(hid_size * 2, 1)  # Output layer for predicting salary

    def forward(self, batch):
        # Unpack the batch
        title, description, categorical_features = batch['Title'], batch['FullDescription'], batch['Categorical']
        
        # Title Encoder
        title_emb = self.title_embedding(title)
        # print(title_emb.shape)

        # title_emb = title_emb.mean(dim=1) # Mean pooling over token embeddings
        
        # title_emb = title_emb.unsqueeze(1)

        # title_emb = title_emb.permute(0, 2, 1)
        # title_emb = self.dropout(title_emb)
        title_emb = self.batch_norm(title_emb)
        title_emb = self.conv_title(title_emb)
        # print(title_emb.shape)
        # title_out = F.relu(self.title_fc(title_emb))  # Pass through fully connected layer
        title_out = F.relu(title_emb)  # Pass through fully connected layer
        # print(title_out.shape)
        title_out = title_out.mean(dim=1)
        # print(title_out.shape)
        # print(title_out.shape)
        # Description Encoder
        desc_emb = self.desc_embedding(description)
        # desc_emb = self.dropout(desc_emb)
        desc_emb = self.batch_norm(desc_emb)


        # desc_emb = desc_emb.mean(dim=1)  # Mean pooling over token embeddings
        # print(desc_emb.shape)
        # desc_emb = desc_emb.unsqueeze(1)

        # title_emb = title_emb.permute(0, 2, 1)
        desc_emb = self.conv_desc(desc_emb)
        # desc_out = F.relu(self.desc_fc(desc_emb)) 
        desc_out = F.relu(desc_emb) # Pass through fully connected layer
        desc_out = desc_out.mean(dim=1)

         # Pass through fully connected layer
        # print(desc_out.shape)

        # Categorical Features Encoder
        cat_out = F.relu(self.cat_fc(categorical_features))  # Pass through fully connected layer
        
        # # Concatenate outputs from all branches
        # # print(title_out.shape, desc_out.shape, cat_out.shape)
        combined = torch.cat([title_out, desc_out, cat_out], dim=1)
        
        # # Common Network
        x = F.relu(self.common_fc1(combined))  # First fully connected layer
        output = self.common_fc2(x).squeeze(-1)  # Output layer (squeeze to remove last dimension)
        
        return output

    @staticmethod
    def load_checkpoint(model, checkpoint_path: str, device) -> None:
        """
        Load a saved model checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded from {checkpoint_path}")
