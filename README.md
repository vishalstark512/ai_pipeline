# ai_pipeline

This is a simple AI pipeline running a Unet segmetation model.
The goal of this pipeline is to not train the Unet model, but use Docker, Nginx and Uwsgi to deploy an AI pipeline that can scale to thousands of users, depending upon the load.
