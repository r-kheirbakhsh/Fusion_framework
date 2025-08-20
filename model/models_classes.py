
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import swin_b, Swin_B_Weights
from pytorch_tabular.models.autoint import AutoIntBackbone, AutoIntConfig
from pytorch_tabular.models.common.heads import blocks 
from pytorch_tabular.models.common.heads import LinearHeadConfig


############################ Initialization functions #####################################

def init_weights(model, init_type="kaiming"):
    '''
    Apply custom initialization to all layers of a model
    Supports: kaiming, xavier, normal
    Works on MLPs, CNNs, and Transformer/Attention blocks
    '''
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "normal":
                nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "normal":
                nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)



def init_classifier_head(layer):
    '''
    Apply Kaiming_unkform initialization of the classifier of a model
    '''
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    


###################################### End of Initialization functions ########################################

########################################## Definition for AutoInt #############################################

class AutoIntModel(nn.Module):
    def __init__(self, config):
        super(AutoIntModel, self).__init__()
        self.config = config
        self.backbone = AutoIntBackbone(config=config)
        self.embedding = self.backbone._build_embedding_layer()
        self.head = self._get_head_from_config() # Assuming binary classification

    def _get_head_from_config(self):
        _head_callable = getattr(blocks, self.config.head)
        return _head_callable(
            in_units=self.backbone.output_dim,
            output_dim=self.config.output_dim,
            config=_head_callable._config_template(**self.config.head_config),
        ) 
                    
    def forward(self, x):
        """
        Forward pass for the AutoInt model.

        Args:
            x (_type:torch.Tensor_): Features of shape (batch_size, num_features)

        Returns:
            x (_type:torch.Tensor_): Output logits of shape (batch_size, num_classes - 1)

        """
        #print(x)
        x_cat = x[:, 0].unsqueeze(1).long()     # sex, categorical
        x_cont = x[:, 1].unsqueeze(1).float()   # age age and the rest of continuous features, continuous  .squeeze(1)
       
        x_dict = {
            "categorical": x_cat,
            "continuous": x_cont
        }
        x = self.embedding(x_dict)
        x = self.backbone(x)
        x = self.head(x)

        return x


# Define AutoIntModel_intermediate Model
class AutoIntModel_intermediate(nn.Module):
    def __init__(self, config):
        super(AutoIntModel_intermediate, self).__init__()
        self.config = config
        self.backbone = AutoIntBackbone(config=config)
        self.embedding = self.backbone._build_embedding_layer()

    def forward(self, x):
        """
        Forward pass for the AutoInt model.

        Args:
            x (_type:torch.Tensor_): Features of shape (batch_size, num_features)

        Returns:
            x (_type:torch.Tensor_): Output logits of shape (batch_size, num_classes - 1)

        """
        #print(x)
        x_cat = x[:, 0].unsqueeze(1).long()     # sex, categorical
        x_cont = x[:, 1].unsqueeze(1).float()   # age, continuous

        x_dict = {
            "categorical": x_cat,
            "continuous": x_cont
        }
        x = self.embedding(x_dict)
        x = self.backbone(x)

        return x


######################################## End of Definition for AutoInt ####################################

################################################# MLP Models ##############################################

# Define MLP_intermediate Model          
class MLP_intermediate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP_intermediate, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.LayerNorm(hidden_size // 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    

# Define MLP_1024_512_256_128 Model
class MLP_1024_512_256_128(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MLP_1024_512_256_128, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.LayerNorm(hidden_size // 2)

        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.LayerNorm(hidden_size // 4)

        self.fc4 = nn.Linear(hidden_size // 4, hidden_size // 8)
        self.bn4 = nn.LayerNorm(hidden_size // 8)

        self.classifier = nn.Linear(hidden_size // 8, num_class)  # Flexible Output Layer
        self.num_class = num_class


    def forward(self, x):
        
        x = x.squeeze(1) 
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)        

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.classifier(x)

        return x

##############################################  End of MLP Models ########################################

############################################# Inter_1_concat Model #######################################

# Define intermediate_1 fusion architecture
class Inter_1_concat (nn.Module):
    def __init__ (self, config):
        super(Inter_1_concat, self).__init__()
        self.config = config

        # Define encoders for each MRI modality
        if 'T1_bias' in self.config.modalities or 'T1c_bias' in self.config.modalities or 'T2_bias' in self.config.modalities or 'FLAIR_bias' in self.config.modalities:
            self.mri_encoders = nn.ModuleDict()

            if self.config.mri_model == 'denseNet121':
                for modality in self.config.modalities:
                    if modality in ['T1_bias', 'T1c_bias', 'T2_bias', 'FLAIR_bias']:
                        # Define MRI encoder
                        self.mri_encoders[modality] = CustomDenseNet121(self.config, in_channels=1 , with_head=0)
            
            elif self.config.mri_model == 'swin_b':
                for modality in self.config.modalities:
                    if modality in ['T1_bias', 'T1c_bias', 'T2_bias', 'FLAIR_bias']:
                        # Define MRI encoder
                        self.mri_encoders[modality] = CustomSwin_b(self.config, in_channels=1 , with_head=0)

            else:
                raise ValueError(f"Unknown MRI model type: {self.config.mri_model}")

        # Define Clinical encoder
        if 'Clinical' in self.config.modalities:
            if self.config.cl_model == 'AutoInt':
                head_config = LinearHeadConfig(
                    layers="",
                    dropout=0.1,  # default
                    initialization=("kaiming"),  # No additional layer in head, just a mapping layer to output_dim                     
                )

                # Initialize AutoInt model with tuned parameters
                model_config = AutoIntConfig(task="classification",  head="LinearHead", attn_dropouts=0.3, attn_embed_dim=4, batch_norm_continuous_input=True, embedding_dim=16, num_attn_blocks=1, num_heads=4)
                model_config.continuous_dim = 1
                model_config.categorical_dim = 1
                model_config.categorical_cardinality = [2]
                model_config.output_cardinality = [2]
                model_config.output_dim = 1  # For binary classification
                model_config.head_config = head_config.__dict__
                self.clinical_encoder = AutoIntModel_intermediate(config=model_config)

            else:
                raise ValueError(f"Unknown Clinical model type: {self.config.cl_model}")
            

        # Total feature size for fusion
        self.total_feature_size = len(self.mri_encoders) * 1024  # DenseNet121 and Swin_b output size
        if 'Clinical' in self.config.modalities:
            if self.config.cl_model == 'AutoInt':
                self.total_feature_size += 8 # self.clinical_encoder.backbone.output_dim for thr tuned model -> 64 with the default structure of AutoInt
            elif self.config.cl_model == 'MLP':
                self.total_feature_size += 16  # MLP outputs 16-dim feature

        # Define classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.total_feature_size, 128),
            nn.LayerNorm(128),  # for small batch size (like 16) BatchNorm1d layer may make training inconsistant 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, self.config.num_class - 1)
        )

        # Initialize classifier weights
        init_weights(self.classifier, init_type="kaiming")


    def forward(self, modalities_input_dict):
        features = []

        for modality, encoder in self.mri_encoders.items():
            x = modalities_input_dict[modality]  # (B, 1, H, W) 
            if self.config.fusion_method == 'early_1_fusion':
                with torch.no_grad():
                    feat = encoder(x)
            else:
                feat = encoder(x)

            features.append(feat)

        if 'Clinical' in self.config.modalities:
            x_c = modalities_input_dict['Clinical']
            if self.config.fusion_method == 'early_1_fusion':
                with torch.no_grad():
                    clinical_feat = self.clinical_encoder(x_c)  # (B, 16)
            else:
                clinical_feat = self.clinical_encoder(x_c)
                       
            features.append(clinical_feat)

        # fusion happens here; concatenation of features
        fused = torch.cat(features, dim=1)
        out = self.classifier(fused)

        return out
    
############################################### End of Inter_1_concat Model ########################################    

################################################## Inter_2_concat Model ############################################

# Define intermediate fusion architecture
class Inter_2_concat (nn.Module):
    def __init__ (self, config):
        super(Inter_2_concat, self).__init__()
        self.config = config

        # Define DenseNet121 encoders for each MRI modality
        if 'T1_bias' in self.config.modalities or 'T1c_bias' in self.config.modalities or 'T2_bias' in self.config.modalities or 'FLAIR_bias' in self.config.modalities:
            
            # Calculate the number of the channel of the image (number of MRI modalities)
            if 'Clinical' in self.config.modalities:
                num_mri_modalities = len(self.config.modalities) - 1  # Exclude Clinical
            else:
                num_mri_modalities = len(self.config.modalities)  

            if self.config.mri_model == 'denseNet121':                   
                # define the MRI encoder
                self.mri_encoder = CustomDenseNet121(self.config, in_channels=num_mri_modalities , with_head=0)

            elif self.config.mri_model == 'swin_b':
                # define the MRI encoder
                self.mri_encoder = CustomSwin_b(self.config, in_channels=num_mri_modalities , with_head=0)

            else:
                raise ValueError(f"Unknown MRI model type: {self.config.mri_model}")

        # Define Clinical encoder
        if 'Clinical' in self.config.modalities:
            if self.config.cl_model == 'AutoInt':
                head_config = LinearHeadConfig(
                    layers="",
                    dropout=0.1,  # default
                    initialization=("kaiming"),  # No additional layer in head, just a mapping layer to output_dim                       
                )
                # Initialize AutoInt model with tuned parameters
                model_config = AutoIntConfig(task="classification",  head="LinearHead", attn_dropouts=0.3, attn_embed_dim=4, batch_norm_continuous_input=True, embedding_dim=16, num_attn_blocks=1, num_heads=4)
                model_config.continuous_dim = 1
                model_config.categorical_dim = 1
                model_config.categorical_cardinality = [2]
                model_config.output_cardinality = [2]
                model_config.output_dim = 1  # For binary classification
                model_config.head_config = head_config.__dict__
                self.clinical_encoder = AutoIntModel_intermediate(config=model_config)

            else:
                raise ValueError(f"Unknown Clinical model type: {self.config.cl_model}")


        # Total feature size for fusion
        if 'T1_bias' in self.config.modalities or 'T1c_bias' in self.config.modalities or 'T2_bias' in self.config.modalities or 'FLAIR_bias' in self.config.modalities:
            self.total_feature_size = 1024  # DenseNet121/Swin_b output size
        else:
            self.total_feature_size = 0

        if 'Clinical' in self.config.modalities:
            if self.config.cl_model == 'AutoInt':
                self.total_feature_size += 8 # self.clinical_encoder.backbone.output_dim for thr tuned model -> 64 with the default structure of AutoInt 
            elif self.config.cl_model == 'MLP':
                self.total_feature_size += 16  # MLP outputs 16-dim feature


        self.classifier = nn.Sequential(
            nn.Linear(self.total_feature_size, 128),
            nn.LayerNorm(128),  # for small batch size (like 16) BatchNorm1d layer may make training inconsistant 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, self.config.num_class - 1)
        )
        
        # Initialize classifier weights
        init_weights(self.classifier, init_type="kaiming")


    def forward(self, modalities_input_dict):
        features = []

        if 'T1_bias' in self.config.modalities or 'T1c_bias' in self.config.modalities or 'T2_bias' in self.config.modalities or 'FLAIR_bias' in self.config.modalities:
            mri_feat = self.mri_encoder(modalities_input_dict['MRI'])
            features.append(mri_feat)

        if 'Clinical' in self.config.modalities:
            clinical_feat = self.clinical_encoder(modalities_input_dict['Clinical'])  # (B, 16)
            features.append(clinical_feat)

        fused = torch.cat(features, dim=1)
        out = self.classifier(fused)

        return out
    
#################################################### End of Inter_2_concat Model #################################################

######################################################## Custom denseNet121 ######################################################

# Define custom denseNet121 that changes the number of the channels of the input image and the head of the model (Identity or number of the classes of the classifier)
class CustomDenseNet121(nn.Module):
    '''' This class defines a custom denseNet121

    '''
    def __init__(self, config, in_channels, with_head):
        '''
        Args:
        config (_type:config_): it contains the configeration of the problem, the ones used in this function:
            pretrained (_type:int_): 0 meants without weights (either non-pretrained or for test), 1 means with weights
            num_classes (_type:int_): the number of classes in the dataset for classification
        in_channels (_type:int_): the number of channels of the input image
        with_head (_type:int_): 1 means with a head for  aclassifier of num_classes classes, 0 means with identity head

        '''  
        super().__init__()

        self.config = config
        self.in_channels = in_channels
        self.with_head = with_head

        if self.config.pretrained == 1:
            model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            model = densenet121(weights=None)
            init_weights(model, init_type="kaiming")

        if in_channels != 3:   
            # Modify first conv layer to accept 1-channel input
            old_conv = model.features[0]
            new_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )

            # Handle pretrained weights
            if self.config.pretrained == 1:
                # Average weights across input channels
                with torch.no_grad():
                    new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            else:
                # Initialize weights for non-pretrained model
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

            model.features[0] = new_conv

        if self.with_head == 1:
            # Modify the final layer for 3 classes (DenseNet uses `classifier` for its final layer)
            if self.config.num_class == 2: 
                model.classifier = nn.Linear(model.classifier.in_features, 1)
            else: 
                model.classifier = nn.Linear(model.classifier.in_features, self.config.num_class)                
            # Initialize weights for the head
            model.classifier.apply(init_classifier_head)

        else:
            # Remove the classifier head
            model.classifier = nn.Identity()  

        self.model = model

    
    def forward(self, x):
        return self.model(x)


############################################ End of custom denseNet121 ############################################

################################################## Custom swin_b ##################################################

# Define custom swin_b that changes the number of the channels of the input image and the head of the model (Identity or number of the classes of the classifier)
class CustomSwin_b(nn.Module):
    '''' This class defines a custom swin_b

    '''
    def __init__(self, config, in_channels, with_head):
        '''
        Args:
        config (_type:config_): it contains the configeration of the problem, the ones used in this function:
            pretrained (_type:int_): 0 meants without weights (either non-pretrained or for test), 1 means with weights
            num_classes (_type:int_): the number of classes in the dataset for classification
        in_channels (_type:int_): the number of channels of the input image
        with_head (_type:int_): 1 means with a head for  aclassifier of num_classes classes, 0 means with identity head

        '''
        super().__init__()

        self.config = config
        self.in_channels = in_channels
        self.with_head = with_head

        # Initialize a single Swin_B model for all MRI modalities
        if self.config.pretrained == 1:
            model = swin_b(weights=Swin_B_Weights.DEFAULT)
        else:
            model = swin_b(weights=None)
            init_weights(model, init_type="kaiming")

        if self.in_channels != 3:   

            # Access the first conv layer inside the patch embedding
            conv_proj = model.features[0][0]  # This is Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))   
            # Create new Conv2d with multiple input channels
            new_conv = nn.Conv2d(
                in_channels=self.in_channels,  # Accept multiple channels for different modalities
                out_channels=conv_proj.out_channels,
                kernel_size=conv_proj.kernel_size,
                stride=conv_proj.stride,
                padding=conv_proj.padding,
                bias=False,
            )   
            # Handle pretrained weights
            if self.config.pretrained == 1:
                with torch.no_grad():
                    new_conv.weight[:] = conv_proj.weight.mean(dim=1, keepdim=True)  # Compute mean across RGB channels and assign mean weight to all additional channels
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')   
                
            # Replace the old conv with the new one
            model.features[0][0] = new_conv

        if self.with_head == 1:
            # Modify the final layer for 3 classes
            if self.config.num_class == 2:
                model.head = nn.Linear(model.head.in_features, 1)
            else:
                model.head = nn.Linear(model.head.in_features, self.config.num_class)
            # Initialize weights for the head
            model.classifier.apply(init_classifier_head)

        else: 
            # Remove the classifier head
            model.head = nn.Identity()
        
        self.model = model


    def forward(self, x):
        return self.model(x)
        

################################################# End of custom swin_b #################################################

############################################## Inter_2_bi_crossattention ###############################################

class CrossAttention(nn.Module):
    '''This is basically a learned, multi-head, modality-to-modality information mixing layer.
        It is doing a one-to-one token cross-attention: Each sample's query vector attends to a single key/value vector from another modality.
        In a full Transformer, queries could attend to multiple tokens (seq_len > 1), but here, both are just 1 token.

    '''
    def __init__(self, dim_q, dim_kv, dim_out, num_heads=4):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads, batch_first=True)
        self.query_proj = nn.Linear(dim_q, dim_out)
        self.key_proj = nn.Linear(dim_kv, dim_out)
        self.value_proj = nn.Linear(dim_kv, dim_out)

    def forward(self, query, key_value):   # query is the modality whose features I want to enhance using information from the key_value modality and key_value is the modality providing the context.
        # query, key_value: (B, dim)
        query = self.query_proj(query).unsqueeze(1)        # (B, 1, dim_out) Unsqueeze: Adds a sequence dimension; seq_len = 1 here because each sample’s query is treated as a single token.
        key = self.key_proj(key_value).unsqueeze(1)        # (B, 1, dim_out)
        value = self.value_proj(key_value).unsqueeze(1)    # (B, 1, dim_out)

        # The _ captures the attention weights, but they’re not used here
        attn_output, _ = self.attn(query, key, value)      # (B, 1, dim_out) Since seq_len=1 for both query and key/value, the attention matrix is trivial (1×1), but multi-head attention still applies the head projections and mixes information.
        return attn_output.squeeze(1)                      # (B, dim_out)


class Inter_2_bidirectional_crossattention(nn.Module):
    def __init__(self, config):
        super(Inter_2_bidirectional_crossattention, self).__init__()
        self.config = config

        # MRI Encoder
        if any(m in config.modalities for m in ['T1_bias', 'T1c_bias', 'T2_bias', 'FLAIR_bias']):
            num_mri_modalities = len(config.modalities) - 1 if 'Clinical' in config.modalities else len(config.modalities)

            if config.mri_model == 'denseNet121':
                self.mri_encoder = CustomDenseNet121(config, in_channels=num_mri_modalities, with_head=0)
            elif config.mri_model == 'swin_b':
                self.mri_encoder = CustomSwin_b(config, in_channels=num_mri_modalities, with_head=0)
            else:
                raise ValueError(f"Unknown MRI model type: {config.mri_model}")

        # Clinical Encoder
        if 'Clinical' in self.config.modalities:
            if self.config.cl_model == 'AutoInt':
                head_config = LinearHeadConfig(
                    layers="",
                    dropout=0.1,  # default
                    initialization=("kaiming"),  # No additional layer in head, just a mapping layer to output_dim                       
                )
                # Initialize AutoInt model
                # model_config = AutoIntConfig(task="classification",  head="LinearHead", attn_dropouts=0.1, attn_embed_dim=32, batch_norm_continuous_input=False, embedding_dim=8, num_attn_blocks=1, num_heads=4)
                model_config = AutoIntConfig(task="classification",  head="LinearHead", attn_dropouts=0.3, attn_embed_dim=4, batch_norm_continuous_input=True, embedding_dim=16, num_attn_blocks=1, num_heads=4)
                model_config.continuous_dim = 1
                model_config.categorical_dim = 1
                model_config.categorical_cardinality = [2]
                model_config.output_cardinality = [2]
                model_config.output_dim = 1  # For binary classification
                model_config.head_config = head_config.__dict__
                self.clinical_encoder = AutoIntModel_intermediate(config=model_config)
                cl_feat_dim = 8  # self.clinical_encoder.backbone.output_dim for thr tuned model -> 64 with the default structure of AutoInt 

            else:
                raise ValueError(f"Unknown Clinical model type: {self.config.cl_model}")

        # Bidirectional Attention 
        if 'Clinical' in config.modalities and any(m in config.modalities for m in ['T1_bias', 'T1c_bias', 'T2_bias', 'FLAIR_bias']):
            mri_feat_dim = 1024  # Feature size from DenseNet121/Swin_b

            self.cross_attn_clinical_to_mri = CrossAttention(cl_feat_dim, mri_feat_dim, 128)
            self.cross_attn_mri_to_clinical = CrossAttention(mri_feat_dim, cl_feat_dim, 128)

            self.total_feature_size = 128 + 128  # concat both
        else:
            self.total_feature_size = cl_feat_dim if 'Clinical' in config.modalities else mri_feat_dim

        # # Classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.total_feature_size, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, config.num_class - 1),  # For binary classification
        # )
        self.classifier = nn.Sequential(
            nn.Linear(self.total_feature_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, self.config.num_class - 1)  # Assuming binary classification
        )

    def forward(self, modalities_input_dict):
        mri_feat = None
        clinical_feat = None

        if 'MRI' in modalities_input_dict:
            mri_feat = self.mri_encoder(modalities_input_dict['MRI'])  # (B, 1024)

        if 'Clinical' in self.config.modalities:
            clinical_feat = self.clinical_encoder(modalities_input_dict['Clinical'])  # (B, dim)

        # Bidirectional Cross-Attention
        if mri_feat is not None and clinical_feat is not None:
            # 1. Clinical attends to MRI
            fused1 = self.cross_attn_clinical_to_mri(clinical_feat, mri_feat)  # (B, 128)

            # 2. MRI attends to Clinical
            fused2 = self.cross_attn_mri_to_clinical(mri_feat, clinical_feat)  # (B, 128)

            # Combine both
            fused = torch.cat([fused1, fused2], dim=1)  # (B, 256)

        elif mri_feat is not None:
            fused = mri_feat
        elif clinical_feat is not None:
            fused = clinical_feat
        else:
            raise ValueError("No input modalities provided.")

        out = self.classifier(fused)
        return out
    
############################################ End of Inter_2_bi_crossattention ###############################

########################################### Inter_2_bi_crossattn_selfattn Model ##################################

class Inter_2_bi_crossattn_selfattn(nn.Module):
    def __init__(self, config):
        super(Inter_2_bi_crossattn_selfattn, self).__init__()
        self.config = config

        # MRI Encoder
        if any(m in config.modalities for m in ['T1_bias', 'T1c_bias', 'T2_bias', 'FLAIR_bias']):
            num_mri_modalities = len(config.modalities) - 1 if 'Clinical' in config.modalities else len(config.modalities)

            if config.mri_model == 'denseNet121':
                self.mri_encoder = CustomDenseNet121(config, in_channels=num_mri_modalities, with_head=0)
            elif config.mri_model == 'swin_b':
                self.mri_encoder = CustomSwin_b(config, in_channels=num_mri_modalities, with_head=0)
            else:
                raise ValueError(f"Unknown MRI model type: {config.mri_model}")

        # Clinical Encoder 
        if 'Clinical' in self.config.modalities:
            if self.config.cl_model == 'AutoInt':
                head_config = LinearHeadConfig(
                    layers="",
                    dropout=0.1,
                    initialization=("kaiming"),                      
                )
                model_config = AutoIntConfig(task="classification",  head="LinearHead", attn_dropouts=0.3, attn_embed_dim=4, batch_norm_continuous_input=True, embedding_dim=16, num_attn_blocks=1, num_heads=4)
                model_config.continuous_dim = 1
                model_config.categorical_dim = 1
                model_config.categorical_cardinality = [2]
                model_config.output_cardinality = [2]
                model_config.output_dim = 1  
                model_config.head_config = head_config.__dict__
                self.clinical_encoder = AutoIntModel_intermediate(config=model_config)
                cl_feat_dim = 8 # self.clinical_encoder.backbone.output_dim for thr tuned model -> 64 with the default structure of AutoInt 

            else:
                raise ValueError(f"Unknown Clinical model type: {self.config.cl_model}")

        # Bidirectional Attention 
        if 'Clinical' in config.modalities and any(m in config.modalities for m in ['T1_bias', 'T1c_bias', 'T2_bias', 'FLAIR_bias']):
            mri_feat_dim = 1024
            self.cross_attn_clinical_to_mri = CrossAttention(cl_feat_dim, mri_feat_dim, 128)
            self.cross_attn_mri_to_clinical = CrossAttention(mri_feat_dim, cl_feat_dim, 128)
            fused_dim = 256
        else:
            fused_dim = cl_feat_dim if 'Clinical' in config.modalities else mri_feat_dim

        # # Feedforward layer after bidirectional cross-attention and before self-attention
        # self.feedforward1 = nn.Sequential(
        #     nn.Linear(fused_dim, fused_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        self.feedforward1 = nn.Sequential(
            nn.Linear(fused_dim, 4 * fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(fused_dim) # Normalizes across the feature dimension for each sample independently, not sensitive to batch size

        # Self-Attention After Feedforward
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,     # split 256 into 4x64 tokens
                nhead=4,
                dropout=0.1,
                batch_first=True   # means attention expects input tensors in shape (batch, seq_len, embed_dim) instead of (seq_len, batch, embed_dim)
            ),
            num_layers=1  # number of the attention layers
        )

        # Feedforward layer after self-attention
        self.feedforward2 = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(fused_dim)

        self.total_feature_size = fused_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.total_feature_size, 128),
            nn.LayerNorm(128),  # for small batch size (like 16) BatchNorm1d layer may make training inconsistant 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, self.config.num_class - 1)
        )

    def forward(self, modalities_input_dict):
        mri_feat = None
        clinical_feat = None

        if 'MRI' in modalities_input_dict:
            mri_feat = self.mri_encoder(modalities_input_dict['MRI'])

        if 'Clinical' in self.config.modalities:
            clinical_feat = self.clinical_encoder(modalities_input_dict['Clinical'])

        if mri_feat is not None and clinical_feat is not None:
            fused1 = self.cross_attn_clinical_to_mri(clinical_feat, mri_feat)
            fused2 = self.cross_attn_mri_to_clinical(mri_feat, clinical_feat)
            fused = torch.cat([fused1, fused2], dim=1)
        elif mri_feat is not None:
            fused = mri_feat
        elif clinical_feat is not None:
            fused = clinical_feat
        else:
            raise ValueError("No input modalities provided.")

        # Feedforward before self-attention
        fused = self.feedforward1(fused)
        fused = self.norm1(fused)

        # Self-Attention Layer 
        fused_seq = fused.view(fused.size(0), 4, 64)   # doing a tensor reshape operation before sending the data into the self-attention block; fused_seq.shape → (batch_size, 4, 64); it takes 256 features and split them into 4 "tokens" each having 64 features
        fused_attended = self.self_attn(fused_seq)
        fused_flat = fused_attended.reshape(fused.size(0), -1)

        # Feedforward after self-attention
        fused_transformed = self.feedforward2(fused_flat)
        fused_transformed = self.norm2(fused_transformed)

        out = self.classifier(fused_transformed)
        return out
####################################### end of Inter_2_bi_crossattn_selfattn ######################################

####################################### Inter_1_selfattn_bi_crossattn Model #######################################

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, seq_len, embed_dim)
        # Self-attention
        attn_out, _ = self.attn(x, x, x) # Q=K=V=x
        x = self.attn_norm(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)
        return x


class MRI_FusionTransformer(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=4, ff_hidden_dim=2048, num_layers=2, dropout=0.1):
        super(MRI_FusionTransformer, self).__init__()
        self.layers = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (B, num_modalities, embed_dim)
        for layer in self.layers:
            x = layer(x)
        return x  # (B, num_modalities, embed_dim)


class CrossAttention(nn.Module):
    """
    Projects query and key/value into dim_out and runs multihead attention.
    Works with inputs of shape (B, dim_in). Returns (B, dim_out).
    """
    def __init__(self, dim_q, dim_kv, dim_out, num_heads=4):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads, batch_first=True) # mri_embed_dim % cross_heads == 0 must hold for nn.MultiheadAttention
        self.query_proj = nn.Linear(dim_q, dim_out)
        self.key_proj = nn.Linear(dim_kv, dim_out)
        self.value_proj = nn.Linear(dim_kv, dim_out)

    def forward(self, query, key_value):
        # query: (B, dim_q), key_value: (B, dim_kv)
        q = self.query_proj(query).unsqueeze(1)    # (B, 1, dim_out)
        k = self.key_proj(key_value).unsqueeze(1)  # (B, 1, dim_out)
        v = self.value_proj(key_value).unsqueeze(1)# (B, 1, dim_out)

        out, _ = self.attn(q, k, v)                # (B, 1, dim_out)
        return out.squeeze(1)                      # (B, dim_out)


# Define Inter_1_selfattn_bi_crossattn fusion architecture
class Inter_1_attn_crossattn_multi(nn.Module):
    def __init__(self, config,
                 mri_embed_dim=1024,
                 mri_self_heads=4,
                 mri_self_ff=2048,
                 mri_self_layers=2,
                 cross_heads=4,
                 cross_ff=2048,
                 cross_layers=2,
                 dropout=0.1):
        """
        - config: your config object (modalities, mri_model, cl_model, num_class, etc.)
        - mri_embed_dim: embedding size output by each MRI encoder (DenseNet121).
        - mri_self_layers: number of stacked self-attention+FF blocks for MRI fusion.
        - cross_layers: number of stacked bidirectional cross-attention + FF blocks.
        """
        super(Inter_1_attn_crossattn_multi, self).__init__()
        self.config = config
        self.mri_embed_dim = mri_embed_dim

        # ---------- MRI encoders ----------
        self.mri_encoders = nn.ModuleDict()
        if self.config.mri_model == 'denseNet121':
            for modality in self.config.modalities:
                if modality in ['T1_bias', 'T1c_bias', 'T2_bias', 'FLAIR_bias']:
                    self.mri_encoders[modality] = CustomDenseNet121(self.config, in_channels=1, with_head=0)
        elif self.config.mri_model == 'swin_b':
            for modality in self.config.modalities:
                if modality in ['T1_bias', 'T1c_bias', 'T2_bias', 'FLAIR_bias']:
                    self.mri_encoders[modality] = CustomSwin_b(self.config, in_channels=1, with_head=0)
        else:
            raise ValueError(f"Unknown MRI model type: {self.config.mri_model}")

        self.num_mri_modalities = len(self.mri_encoders)

        # ---------- MRI fusion transformer (stacked self-attention + FF) ----------
        # mri_embed_dim % mri_self_heads == 0 must hold 
        self.mri_fusion_transformer = MRI_FusionTransformer(
            embed_dim=self.mri_embed_dim,
            num_heads=mri_self_heads,
            ff_hidden_dim=mri_self_ff,
            num_layers=mri_self_layers,
            dropout=dropout
        )

        # ---------- Clinical encoder (keep your AutoInt or MLP as-is) ----------
        if 'Clinical' in self.config.modalities:
            if self.config.cl_model == 'AutoInt':
                # build the same AutoIntModel_intermediate setup you used previously
                head_config = LinearHeadConfig(layers="", dropout=0.1, initialization=("kaiming"))
                model_config = AutoIntConfig(task="classification", head="LinearHead",
                                            attn_dropouts=0.3, attn_embed_dim=4,
                                            batch_norm_continuous_input=True,
                                            embedding_dim=16, num_attn_blocks=1, num_heads=4)
                model_config.continuous_dim = 1
                model_config.categorical_dim = 1
                model_config.categorical_cardinality = [2]
                model_config.output_cardinality = [2]
                model_config.output_dim = 1
                model_config.head_config = head_config.__dict__
                self.clinical_encoder = AutoIntModel_intermediate(config=model_config)
                self.clinical_output_dim = 8 # self.clinical_encoder.backbone.output_dim for thr tuned model -> 64 with the default structure of AutoInt 
              
            else:
                raise ValueError(f"Unknown Clinical model type: {self.config.cl_model}")
        else:
            self.clinical_encoder = None
            self.clinical_output_dim = 0

        # ---------- project clinical to common embedding dim (so we can stack cross layers with residuals) ----------
        if self.clinical_encoder is not None:
            self.clinical_project = nn.Linear(self.clinical_output_dim, self.mri_embed_dim)
            self.clinical_project_norm = nn.LayerNorm(self.mri_embed_dim)
        else:
            self.clinical_project = None

        # ---------- Cross-attention stacked blocks ----------
        # For simplicity we use the same dim_out = mri_embed_dim for all cross attention blocks.
        self.cross_layers = nn.ModuleList()
        for _ in range(cross_layers):   
            block = nn.ModuleDict({
                # cross attention modules (both directions)
                'cl_to_mri_attn': CrossAttention(dim_q=self.mri_embed_dim, dim_kv=self.mri_embed_dim, dim_out=self.mri_embed_dim, num_heads=cross_heads),
                'mri_to_cl_attn': CrossAttention(dim_q=self.mri_embed_dim, dim_kv=self.mri_embed_dim, dim_out=self.mri_embed_dim, num_heads=cross_heads),
                # norms and feedforwards for each branch
                'mri_attn_norm': nn.LayerNorm(self.mri_embed_dim),
                'cl_attn_norm' : nn.LayerNorm(self.mri_embed_dim),
                'mri_ff' : nn.Sequential(
                    nn.Linear(self.mri_embed_dim, cross_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(cross_ff, self.mri_embed_dim),
                    nn.Dropout(dropout)
                ),
                'cl_ff' : nn.Sequential(
                    nn.Linear(self.mri_embed_dim, cross_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(cross_ff, self.mri_embed_dim),
                    nn.Dropout(dropout)
                ),
                'mri_ff_norm': nn.LayerNorm(self.mri_embed_dim),
                'cl_ff_norm' : nn.LayerNorm(self.mri_embed_dim)
            })
            self.cross_layers.append(block)

        # ---------- Self-attention block after Cross-attention block ----------
        self.post_cross_selfattn = SelfAttentionBlock(embed_dim=self.mri_embed_dim, num_heads=4, ff_hidden_dim=2048 , dropout=0.1)

        # ---------- final classifier ----------
        self.classifier = nn.Sequential(
            nn.Linear(self.mri_embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.config.num_class - 1)
        )

    # Definition of forwaed funvtion
    def forward(self, modalities_input_dict):

        # ----- 1) extract MRI features per modality -----
        mri_feats = []
        for modality, encoder in self.mri_encoders.items():
            x = modalities_input_dict[modality]   # (B, 1, H, W)
            feat = encoder(x)                     # expected (B, mri_embed_dim)
            mri_feats.append(feat)
        
        mri_feats = torch.stack(mri_feats, dim=1) # (B, num_mri_modalities, embed_dim)

        # ----- 2) MRI fusion: stacked self-attention + MLP -----
        mri_attended_tokens = self.mri_fusion_transformer(mri_feats)  # (B, num_mri_modalities, embed_dim)

        # ----- 3) mean pool tokens to get single MRI vector -----
        mri_vec = mri_attended_tokens.mean(dim=1)  # (B, embed_dim)

        # ----- 4) get clinical features and project to common dim -----
        if self.clinical_encoder is not None:
            cl_feat = self.clinical_encoder(modalities_input_dict['Clinical'])  # (B, clinical_output_dim)
            cl_proj = self.clinical_project(cl_feat)                           # (B, embed_dim)
            cl_proj = self.clinical_project_norm(cl_proj)
            cl_vec = cl_proj
        else:
            # if no clinical, we still allow cross layers to be skipped
            cl_vec = None

        # ----- 5) stacked bidirectional cross-attention -----
        if cl_vec is not None:
            # both mri_vec and cl_vec are (B, embed_dim)
            m = mri_vec
            c = cl_vec
            for layer in self.cross_layers:
                # cross-attention updates (note: layer['cl_to_mri_attn'] expects (B,dim) inputs)
                m_update = layer['cl_to_mri_attn'](c, m)   # clinical -> mri (B, embed_dim)
                c_update = layer['mri_to_cl_attn'](m, c)   # mri -> clinical (B, embed_dim)

                # residual + norm
                m = layer['mri_attn_norm'](m + m_update)
                c = layer['cl_attn_norm'](c + c_update)

                # feedforward + residual + norm
                m_ff = layer['mri_ff'](m)
                c_ff = layer['cl_ff'](c)
                m = layer['mri_ff_norm'](m + m_ff)
                c = layer['cl_ff_norm'](c + c_ff)

            # Put tokens in a list to allow for future modalities
            tokens = [m, c]  # you could append more here if needed
        else:
            tokens = [mri_vec]

        # ----- 6) self-attention over all tokens -----
        tokens = torch.stack(tokens, dim=1)           # (B, num_tokens, embed_dim)
        tokens = self.post_cross_selfattn(tokens)     # (B, num_tokens, embed_dim)

        # ----- 7) flatten for classifier -----
        fused = tokens.flatten(start_dim=1)           # (B, num_tokens * embed_dim)

        # ----- 8) classify -----
        out = self.classifier(fused)
        return out





  


