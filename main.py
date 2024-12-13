from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_caching import Cache
import spacy
import os
import toml
import time
from multiprocessing import Pool, cpu_count
from src.image_processor import ImageProcessorFactory, logger  # Import your ImageProcessorFactory
from src.json_cleaning import *  # Import your JSON cleaning module
import win32security
import win32con
from configparser import ConfigParser

# Load configuration
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config', 'config.toml'))
config = toml.load(config_path)
nlp_model_path = config['paths']['nlp_model']

# Initialize Flask app and caching
app = Flask(__name__)
CORS(app)
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 500
cache = Cache(app)

# Load NLP model globally at app startup
nlp = spacy.load(nlp_model_path)

# Define number of processors to use
num_processors = 12
text_noise = "![]+{};'\"\\,<>.?#$%^*_~'—|"

def GetConfigSetting(obj, name):
    config_object = ConfigParser()
    config_object.read("config.ini")
    try:
        return config_object[obj][name]
    except KeyError:
        logger.error(f"Config setting '{name}' not found in section '{obj}'.")
        raise

serverconfigSetting = GetConfigSetting("SERVERCONFIG_SETTING", "SERVERCONFIG_SETTING")

# Helper function for impersonation and image processing
def impersonate_and_process_image(image_path):
    try:
        connection_string = GetConfigSetting(serverconfigSetting, "connectionString")
        domain, username, pw = connection_string.split(";")
        
        logger.info(f"Impersonating user: {domain}\\{username}")
        handle = win32security.LogonUser(
            username, domain, pw, win32con.LOGON32_LOGON_INTERACTIVE, win32con.LOGON32_PROVIDER_DEFAULT
        )
        win32security.ImpersonateLoggedOnUser(handle)

        processor = ImageProcessorFactory.create(nlp, text_noise)
        result = processor.process_single_image(image_path)

        return result
    except Exception as e:
        logger.error(f"Impersonation error: {str(e)}")
        return {"error": str(e)}
    finally:
        if handle:
            win32security.RevertToSelf()
            handle.Close()

# Multiprocessing helper to process single images
def process_single_image(args):
    text_noise, image_path = args
    return {
        "image_filename": os.path.basename(image_path),
        "key_value_pairs": ImageProcessorFactory.create(nlp, text_noise).process_single_image(image_path)
    }

# Helper to get all images in a folder
def get_all_images_from_folder(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

@app.route('/process_images', methods=["GET"])
@cache.cached(query_string=True)
def process_images():
    impersonate_flag = request.args.get('imp', default='0')
    path = request.args.get('folderPath')

    if not path:
        return jsonify({"error": "folderPath parameter is missing"}), 400

    if not os.path.exists(path):
        return jsonify({"error": "Invalid path"}), 400

    start_time = time.time()

    try:
        if impersonate_flag == '1':
            if os.path.isdir(path):
                # Process each image in folder individually with impersonation
                image_paths = get_all_images_from_folder(path)

                if not image_paths:
                    return jsonify({"error": "No valid image files found"}), 400

                results = []
                with Pool(num_processors) as pool:
                    results = pool.map(impersonate_and_process_image, [(img) for img in image_paths])

                total_time = time.time() - start_time
                logger.info(f"Processed {len(image_paths)} images in {total_time:.2f} seconds")

                return jsonify({"images": results})

            elif os.path.isfile(path):
                folder_name = os.path.basename(os.path.dirname(path))
                result = impersonate_and_process_image(path, text_noise)
                
                total_time = time.time() - start_time
                logger.info(f"Processed single image {path} in {total_time:.2f} seconds")

                return jsonify({
                    folder_name: [{
                        "image_filename": os.path.basename(path),
                        "key_value_pairs": result
                    }]
                })

        else:
            if os.path.isdir(path):
                image_paths = get_all_images_from_folder(path)

                if not image_paths:
                    return jsonify({"error": "No valid image files found"}), 400

                with Pool(num_processors) as pool:
                    results = pool.map(process_single_image, [(text_noise, img) for img in image_paths])

                total_time = time.time() - start_time
                logger.info(f"Processed {len(image_paths)} images in {total_time:.2f} seconds")

                return jsonify({"processed_images": results})

            elif os.path.isfile(path):
                folder_name = os.path.basename(os.path.dirname(path))
                image_result = ImageProcessorFactory.create(nlp, text_noise).process_single_image(path)

                total_time = time.time() - start_time
                logger.info(f"Processed single image {path} in {total_time:.2f} seconds")

                return jsonify({
                    folder_name: [{
                        "image_filename": os.path.basename(path),
                        "key_value_pairs": image_result
                    }]
                })

    except Exception as e:
        logger.error(f"Error in process_images: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
