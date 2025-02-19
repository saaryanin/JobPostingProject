Here goes all the code that generates the server.
You should build endpoints for receiving requests. Make sure to include at least 3 endpoints:

1. Request a single prediction - single data point is provided to the model and you should return a prediction.
2. Request a batch of predictions - some data structure (i.e list) containing multiple datapoints
and you should return the prediction for each.
3. Health - an endpoint that you can query just to make sure the model is running. It can return anything,
usually something like 'OK'.

Make sure the appropriate verb is used (get/post/patch/put/delete), its named reasonably and doesn't crash easily
(if for example it gets the wrong input)

