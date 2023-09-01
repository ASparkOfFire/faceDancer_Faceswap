import logging
import os
import shutil
import tempfile
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
from fastapi import FastAPI, UploadFile, File, Response
from networks.layers import AdaIN, AdaptiveAttention
from retinaface.models import *
from utils.options import FaceDancerOptions
from utils.swap_func import run_inference

logging.getLogger().setLevel(logging.ERROR)

opt = FaceDancerOptions().parse()

if len(tf.config.list_physical_devices('GPU')) != 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')

print('\nInitializing FaceDancer...')
RetinaFace = load_model(opt.retina_path, compile=False, custom_objects={"FPN": FPN,
                                                                      "SSH": SSH,
                                                                      "BboxHead": BboxHead,
                                                                      "LandmarkHead": LandmarkHead,
                                                                      "ClassHead": ClassHead})

ArcFace = load_model(opt.arcface_path, compile=False)

G = load_model(opt.facedancer_path, compile=False, custom_objects={"AdaIN": AdaIN,
                                                               "AdaptiveAttention": AdaptiveAttention,
                                                               "InstanceNormalization": InstanceNormalization})
G.summary()

app = FastAPI()

@app.get("/")
def test_api():
    return {"Hello": "World"}

@app.post("/faceswap")
async def generate(source: UploadFile = File(...), target: UploadFile = File(...)):
    os.makedirs('./results', exist_ok=True)
    os.system('rm -rf ./results/*')

    # Create temporary directories to store the uploaded images
    with tempfile.TemporaryDirectory() as tmp_content_dir, tempfile.TemporaryDirectory() as tmp_target_dir, tempfile.TemporaryDirectory() as tmp_output_dir:
        # Save content image file
        input_image_file_path1 = os.path.join(tmp_content_dir, source.filename)
        with open(input_image_file_path1, "wb") as buffer:
            shutil.copyfileobj(source.file, buffer)

        # Save target image file
        input_image_file_path2 = os.path.join(tmp_target_dir, target.filename)
        with open(input_image_file_path2, "wb") as buffer:
            shutil.copyfileobj(target.file, buffer)

        print('\nProcessing images...')
        
        # Move the result image to the temporary output directory
        filename = next(tempfile._get_candidate_names()) + ".png"
        output_file = os.path.join(tmp_output_dir, filename)
        
        run_inference(opt, input_image_file_path1, input_image_file_path2, RetinaFace, ArcFace, G, output_file)
        
        # Read the result image data
        with open(output_file, "rb") as result_image_file:
            result_image_data = result_image_file.read()
        

        print('\nDone! Result saved to {}'.format(output_file))
        
        # Return the result image as a downloadable attachment
        response = Response(content=result_image_data)
        response.headers["Content-Disposition"] = "attachment; filename={target_image.filename}"
        response.headers["Content-Type"] = "image/jpeg"
        return response
    
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)

