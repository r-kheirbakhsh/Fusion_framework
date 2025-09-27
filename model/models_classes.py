
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from pytorch_tabular.models.autoint import AutoIntBackbone, AutoIntConfig
from pytorch_tabular.models.common.heads import blocks 
from pytorch_tabular.models.common.heads import LinearHeadConfig


def set_seed(config):
    '''This function sets random seeds for reproducibility
    
    '''
    # Set seeds for PyTorch to ensure consistency across runs
    torch.manual_seed(config.seed)

    # Using a GPU, make operations deterministic by setting:
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

############################ Initialization functions #####################################

def init_weights(config, model, init_type="kaiming"):
    '''
    Apply custom initialization to all layers of a model
    Supports: kaiming, xavier, normal
    Works on MLPs, CNNs, and Transformer/Attention blocks
    '''
    # Set the seed
    set_seed(config)

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



def init_classifier_head(config, layer):
    '''
    Apply Kaiming_unkform initialization of the classifier of a model
    '''
    # Set the seed
    set_seed(config)

    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    

###################################### End of Initialization functions ########################################

########################################## Definition of AutoInt #############################################

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


######################################## End of Definition of AutoInt ####################################

################################################# MLP Model ##############################################

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

##############################################  End of MLP Model ########################################

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
        init_weights(self.config, self.classifier, init_type="kaiming")


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

        # Define encoders for each MRI modality
        # if 'T1_bias' in self.config.modalities or 'T1c_bias' in self.config.modalities or 'T2_bias' in self.config.modalities or 'FLAIR_bias' in self.config.modalities:
        if any(m in self.modalities for m in ['T1_bias','T1c_bias','T2_bias','FLAIR_bias']):

            # Calculate the number of the channel of the image (number of MRI modalities)
            if 'Clinical' in self.config.modalities:
                num_mri_modalities = len(self.config.modalities) - 1  # Exclude Clinical
            else:
                num_mri_modalities = len(self.config.modalities)  

            if self.config.mri_model == 'denseNet121':                   
                # define the MRI encoder
                self.mri_encoder = CustomDenseNet121(self.config, in_channels=num_mri_modalities , with_head=0)

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
        init_weights(self.config, self.classifier, init_type="kaiming")


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
            init_classifier_head(self.config, model.classifier)

        else:
            # Remove the classifier head
            model.classifier = nn.Identity()  

        self.model = model

    
    def forward(self, x):
        return self.model(x)


############################################ End of custom denseNet121 ############################################

################################################### Modality weighting Block ###################################################


class ModalityAttention(nn.Module):
    """
    Modality Attention with shared attention network.

    """
    def __init__(self, config, input_dims, hidden_dim=256):
        super(ModalityAttention, self).__init__()
        self.config = config

        # Project each modality into same hidden_dim space
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim) for in_dim in input_dims
        ])
        # Initialize projections weights
        for proj in self.projections:
            init_weights(self.config, proj, init_type="kaiming")

        # Shared attention scorer
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # scalar score per modality
        )
        # Initialize attention networks weights
        init_weights(self.config, self.attn, init_type="kaiming")

    def forward(self, feats):  
        # feats = list of modality feature tensors, each (B, in_dim)
        proj_feats = [proj(f) for proj, f in zip(self.projections, feats)]  
        proj_feats = torch.stack(proj_feats, dim=1)  # (B, M, hidden_dim)

        scores = self.attn(proj_feats)               # (B, M, 1)
        attn_weights = torch.softmax(scores, dim=1)  # (B, M, 1)

        fused = torch.sum(attn_weights * proj_feats, dim=1)  # (B, hidden_dim)
        return fused, attn_weights


################################################ End of Modality weighting Block ################################################

############################################### Inter_1_concat_attn Model ###############################################

class Inter_1_concat_attn (nn.Module):
    def __init__ (self, config):
        super(Inter_1_concat_attn, self).__init__()
        self.config = config
        self.modalities = config.modalities

        # ----- Encoders -----
        input_dims = []

        # Define encoders for each MRI modality
        if any(m in self.modalities for m in ['T1_bias','T1c_bias','T2_bias','FLAIR_bias']):
            self.mri_encoders = nn.ModuleDict()

            if self.config.mri_model == 'denseNet121':
                for modality in self.modalities:
                    if modality in ['T1_bias', 'T1c_bias', 'T2_bias', 'FLAIR_bias']:
                        # Define MRI encoder
                        self.mri_encoders[modality] = CustomDenseNet121(self.config, in_channels=1 , with_head=0)
                        input_dims.append(1024)

            else:
                raise ValueError(f"Unknown MRI model type: {self.config.mri_model}")

        # Define Clinical encoder
        if 'Clinical' in self.modalities:
            if self.config.cl_model == 'AutoInt':
                head_config = LinearHeadConfig(
                    layers="",
                    dropout=0.1,  # default
                    initialization=("kaiming"),  # No additional layer in head, just a mapping layer to output_dim                     
                )

                # Initialize AutoInt model with tuned parameters
                model_config = AutoIntConfig(task="classification", head="LinearHead", attn_dropouts=0.3, attn_embed_dim=4, batch_norm_continuous_input=True, embedding_dim=16, num_attn_blocks=1, num_heads=4)
                model_config.continuous_dim = 1
                model_config.categorical_dim = 1
                model_config.categorical_cardinality = [2]
                model_config.output_cardinality = [2]
                model_config.output_dim = 1  # For binary classification
                model_config.head_config = head_config.__dict__
                self.clinical_encoder = AutoIntModel_intermediate(config=model_config)

                input_dims.append(8)  # output dim of AutoInt

            else:
                raise ValueError(f"Unknown Clinical model type: {self.config.cl_model}")  

        # ----- Attention Fusion -----
        self.modality_attention = ModalityAttention(self.config, input_dims, hidden_dim=256)


        # ----- Classifier -----
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
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
        init_weights(self.config, self.classifier, init_type="kaiming")


    def forward(self, modalities_input_dict):
        features = []

        # MRI feature
        for modality, encoder in self.mri_encoders.items():
            x = modalities_input_dict[modality]  # (B, 1, H, W)
            feat = encoder(x)
            features.append(feat)

        # Clinical feature
        if 'Clinical' in self.modalities:
            clinical_feat = self.clinical_encoder(modalities_input_dict['Clinical'])                       
            features.append(clinical_feat)

        # Attention fusion
        fused, attn_weights = self.modality_attention(features)
        # fused, gates = self.gated_modality_attention(features)

        # Classify
        out = self.classifier(fused)
        return out, attn_weights
        #return out, gates

############################################### End of Inter_1_concat_attn Model ####################################################