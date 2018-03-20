# mnist_tsne
t-sne visualization of mnist images when feature is represented by raw pixels and cnn learned feature

# something to say
- the training code is from [pytorch mnist example](https://github.com/pytorch/examples/tree/master/mnist). The accuracy is 98% when use the original code, when bn is used in convolution and fully connected layer, the accuracy is 99. The training code here is with bn.
- the code for t-sne visualization is from [danielfrg/tsne](https://github.com/danielfrg/tsne)
- you can find the original mnist train raw data(60000x784), lable(60000x1), cnn learned feature(60000x50), t-sne generated feature(60000x2) for raw data and cnn learned feature, trained model in Baidu Pan or Google Drive
- tsne_vis.ipynb is used to do tsne and visualization

# visualization
![Alt text](train/data_2d.png? raw=true "bingbing")
[t-sne of raw image pixel](train/data_2d.png)
[t-sne of cnn learned feature](train/output_2d.png)

