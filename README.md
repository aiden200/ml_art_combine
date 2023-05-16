## Check out my [Website]("https://www.aidenwchang.com/)


## How to run the code

Run the code typing in:
```
python3 generate_art.py -c [CONTENT FILEPATH] -s [STYLE FILEPATH]
```

Where the content file is the original image, and the style file contains the artwork image. The generated image will generate an image similar to the content file, with the style of the style file. Please allow some time for the model to train. You can locate the generated models in :
```
/content
```

### Acknowledgements
Thank you to Coursera and Andrew Ng for the helpful lectures for learning the content!


## Building a NST

A NST (Neural Style Transfer) is a technique where we blend the content image with the style of the artwork. 

The key idea behind this is that the shallow hidden layers of a CNN detects low level features such as horizontal contrast patterns while the deep hidden layers detect high features such as face shapes and different types of classes. 

So given a generated image $G$, we can find the loss function $J(G)$ with: <br>
- $J(G) = \alpha J_{content}(h_C,h_G) + \beta J_{style}(h_S,h_G)$

- $\alpha J_{content}(h_C,h_G)$ represents the content loss function given the hidden layers of the content image denoted by $h_C$ and the hidden layers of the generated image denoted by $h_G$. 

- $\beta J_{style}(h_S,h_G)$ represents the style loss function given the hidden layers of the style image denoted by $h_S$ and the hidden layers of the generated image denoted by $h_G$. 

- $alpha$ and $beta$ are just arbitual constants that weigh each component. 

We will use a pre-train model: VGG network to apply transfer learning. Specifically, we will be using VGG-19, a 19 layer VGG network. 

We will first talk about the cost function $J_{content}(h_C,h_G)$. <br>
The cost function can be computed as follows:<br>
- $J_{content}(h_C,h_G)=1/(4*H*W*D) \sum (h_C-h_G)^2$
- $H, W, D$ represents the height, width, and depth of the hidden layer $h_C$. 
- The dimensions of $h_C$ is equivalent to $h_G$, since both images are fed through the same neural network.


Since we want the generated image to look like the content image, we will be selecting the hiden layer at the very end of the neural network for comparison. This will allow the model to learn that high level features needs to be preserved. 


