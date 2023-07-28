# Bidirectional VAE
### A product of my Summer 2023 internship at NYU Courant

Bidirectional Variational AutoEncoder(BVAE)

Research Paper

NYU Courant

Jul 28, 2023

Paul Nieuwerburgh

[See all runs on WandB](https://wandb.ai/paul-nieuwerburgh/BVAE/workspace?workspace=user-paul-nieuwerburgh)

[See the code on GitHub](https://github.com/paulcode123/Bidirectional-VAE)

[View This Paper on Google Docs](https://docs.google.com/document/d/1C0c5Q0zpZQDbd_Lu6MJwGMYE_JJLjOwVI95NSHPJ-jU/edit?usp=sharing)
## Abstract
Robot Learning, along with other low-data applications of machine learning, is plagued with one main issue: its inability to generalize well in other environments and states. This is because Image processing algorithms cannot reliably pick out the important features in the input data. Other, high-data ML applications, such as LLMs, can create a complete latent distribution that can accurately handle any input. VAEs are a good tool to use for this, as they can create a latent distribution that is both specific to the input, and also continuous. 

To incorporate this method in robot learning, we can use one key advantage: when training models, accurate labels must be available. By utilizing the labels in the dataset, we can guide a VAE into creating a continuous latent space despite the sparse population of images represented by it.

The network we derived was one that intertwined all of the core elements of self-supervised learning, while ultimately guiding it by supervised methods. Our model can be thought of as a standard VAE working in tandem with an inverted VAE. 

## Specifics
The crucial part of this concept is that the encoders and decoders are the same between the two networks. By using the predefined labels as input to an inverse of the standard model, we can force the encoder and decoder to be compatible with both the real image/label as well as its representation forcing the representation closer to the real value. Supplemented by incorporating supervised losses into the loss function, the network will find the most accurate way to minimize reconstruction loss while modeling the latent space after the desired distribution.

By intersecting self-supervised and supervised learning we hope to inherit the best of both worlds. BVAE outperforms a neural net in the efficiency that it backpropages the loss and in its ability to represent a distribution of data. This allows the BVAE to generalize more effectively. It beats a standard VAE in the amount of training data it requires.

Another possibility is to use a standard VAE to create an optimal representation, and then use a neural net to map that representation onto the action. However, this approach comes with its own disadvantages. Firstly, using an additional model will create a new source of loss, one that is not backpropogated in the same step as the VAE model, creating misalignment between the models. Secondly, the advantage in using a VAE lies in its ability to accurately create and reconstruct new data points(it’s good at generalizing), and although this may be reflected in the representation, the neural net might not be able to translate that into a new action. All in all, BVAE is the best way to use self-supervised learning in an application where labels are available.

## Network Architecture
![BVAE Structure](<Screenshot (1).png>)
**Diagram 1** shows two VAEs: Network 1 takes as input the image and creates a lower-dimension representation of the image. That representation will be the action, or future state of the robot. Network 2 takes as input an action and manipulates it to form an image.
![BVAE Logistics](<Screenshot (3).png>)
**Diagram 2** shows the shapes of the data at each point and the layers used.

## Training
In order to measure the ability of BVAEs to generalize well, only 8 trajectories were used in the train dataset. The images and actions from the training data are fed into their respective VAEs. The loss consists of the following elements in some arrangement.
Components of loss:
- Reconstruction loss(for each VAE)
- Supervised loss of action(representation of image vs. action)
- Supervised loss of image(representation of action vs. image)

## Testing
As only the image would be given during testing, only network 1 would be used to create an action. The reason for network 2 is to quicken training and to create a more complete distribution in the latent space of Network 1.

## Findings
To prove that BVAE was the best application of supervised learning given our data, we wanted to prove the following:
1. BVAE is accurate: The model minimizes the supervised and reconstruction losses better than other models
2. BVAE is versatile: The model performs better on the validation dataset than other models
3. BVAE is generative: The model’s latent parameters can be manipulated in order to create new, realistic images

We wanted to show that BVAE could compete with the most common supervised learning method: Behavior Cloning, so we tested it against that. We also wanted to show the benefits of our dual-network model compared to a standard VAE, so we added a linear layer to the latent representation of the VAE to make a VAE+NN, and used that as another source of comparison for our BVAE. To eliminate variables in this experiment, we used the same encoder and decoder for each comparative model, only changing the loss function.
![Image Supervised Loss Graph](<W&B Chart 7_28_2023, 12_58_37 PM.png>)
This graph shows how our networks performed, as measured by the loss between the real action and the latent action. This is our main performance metric, as it evaluates the encoder, which will be used in testing. As predicted, the BVAE did only slightly worse than the NN, and better than the VAE+NN in training. With the validation dataset, the BVAE did slightly better than the NN, and about 460x better than the VAE+NN. [Out of the bounds of this graph] This shows that the BVAE can create a better distribution of data than the NN, and we expect this divide to increase with more training and a larger dataset. Using a VAE, our approach is clearly more suited for supervised learning uses, because even though the VAE+NN did alright during training, the NN was vastly unequipped to deal with the new data.

Note: all loss values are measured in standard deviations of the distribution from the training dataset.
![Image Reconstruction Loss Graph](<W&B Chart 7_28_2023, 1_17_48 PM.png>)
Our model also achieved relatively high reconstruction accuracy, given that our latent space was a 1d array of 7 values. In training, the BVAE did about the same as the VAE+NN, which is impressive given that the VAE+NN is optimized to just give the best representation possible, whereas the BVAE must also model the latent space after the real action. Again, the BVAE had an advantage in testing, doing better than the VAE+NN. 
![Model Losses](<Screenshot (4).png>)
The BVAE was very good at producing an action given an image, much better than it was at producing an image given an action. This is logical, as vastly compressing data is easier than vastly expanding it. Additionally, this held true whether the input image to the encoder was a real image or an output of the decoder. In other words, the BVAE was well suited to handle real data, and it could also take self-produced data. As an example of this, the Action Reconstruction loss was 0.1, showing two things: the decoder was good enough to convey the original action, and the encoder was able to take the output of the decoder and reconstruct the action. The model was also good enough to perceive depth, as shown in the diagram to the right. The boxes drawn around the cups in each of the image pairs are the same size, and quite accurately confine both cups. Where in the frame the cups are located is also well preserved, with the boxes in the second image pair lining up exactly, and those in the third pair being not far off. It’s quite clear that the image in the area of focus(the cup) is more detailed than the monotonous gray of the background. In the bottom two reconstructions, the swirls and patterns on the cup can be seen. This shows that the supervised aspect of the model was able to guide the distribution towards latent variables that were important for determining the action.
![Reconstructions](<Screenshot (5).png>)
![Latent Sampling](<Screenshot (6).png>)
The latent variables can also be directly manipulated. We can take the original image in the third pair and get its logits, as shown in the diagram to the right. We can then change the second coordinate, responsible for measuring the distance along the axis pointing towards the microwave, increasing it to 1. Remarkably, this in turn makes the decoder assume that the gripper is much closer to the microwave, and creates a visibly larger cup to represent that change.

In conclusion, the BVAE was able to meet our requirements, surpassing Behavior Cloning and VAE+NN in its accuracy, its ability to generalize, and its ability to model a latent space after the label distribution. With more training and data, this model could have a high performance and high consistency in manipulating latent variables to produce new samples.
