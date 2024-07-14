import os.path
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from inference_audio import *
from django.core.files.storage import FileSystemStorage
from inference_image import *
from inference import *
from text_classification_inference import *
import cv2
from zipfile import ZipFile


@csrf_exempt
def export_function(request):
    context = {}
    if request.method == 'POST':
        try:
            main_flag = request.POST.get("main_flag")

            if main_flag == 'image_classification':

                test_file = request.FILES.get("test_file")
                test_file_save_path = f"static/test_file.jpg"
                try:
                    os.remove(test_file_save_path)
                except:
                    pass
                fs = FileSystemStorage()
                fs.save(test_file_save_path, test_file)

                classes_file = request.FILES.get("classes_file")
                classes_file_save_path = f"static/classes.json"
                try:
                    os.remove(classes_file_save_path)
                except:
                    pass
                fs = FileSystemStorage()
                fs.save(classes_file_save_path, classes_file)

                models = request.FILES.get("model")
                model_save_path = f"static/image_classification_model.pkl"
                try:
                    os.remove(model_save_path)
                except:
                    pass
                fs = FileSystemStorage()
                fs.save(model_save_path, models)

                class_res, confidences, classes_names = predict_image_class(test_file_save_path, classes_file_save_path)
                context["status"] = "success"
                context["Predicted Class"] = class_res
                context["Confidences"] = confidences
                context["All Classes"] = classes_names
            elif main_flag == 'pose_classification':

                test_file = request.FILES.get("test_file")
                test_file_save_path = f"static/test_file.jpg"
                try:
                    os.remove(test_file_save_path)
                except:
                    pass
                fs = FileSystemStorage()
                fs.save(test_file_save_path, test_file)

                classes_file = request.FILES.get("data_file")
                classes_file_save_path = f"static/data.csv"
                try:
                    os.remove(classes_file_save_path)
                except:
                    pass
                fs = FileSystemStorage()
                fs.save(classes_file_save_path, classes_file)

                models = request.FILES.get("model")
                model_save_path = f"static/final_model.h5"
                try:
                    os.remove(model_save_path)
                except:
                    pass
                fs = FileSystemStorage()
                fs.save(model_save_path, models)

                cls, conf, classes = test_image(classes_file_save_path, cv2.imread(test_file_save_path))
                conf = [float(cnf) for cnf in conf]
                context["status"] = "success"
                context["Predicted Class"] = cls
                context["Confidences"] = conf

            elif main_flag == 'audio_classification':

                test_file = request.FILES.get("test_file")
                test_file_save_path = f"static/test_file.wav"
                try:
                    os.remove(test_file_save_path)
                except:
                    pass
                fs = FileSystemStorage()
                fs.save(test_file_save_path, test_file)

                try:
                    os.makedirs('static/dataset')
                except:
                    pass
                classes_file = request.FILES.getlist("data_file")

                classes_file_save_path = f"static/dataset/classes.txt"
                data_file_save_path = f"static/dataset/data.csv"
                try:
                    os.remove(classes_file_save_path)
                except:
                    pass
                try:
                    os.remove(data_file_save_path)
                except:
                    pass
                for f in classes_file:
                    if f.name.endswith(".csv"):
                        fs = FileSystemStorage()
                        fs.save(data_file_save_path, f)
                    elif f.name.endswith(".txt"):
                        fs = FileSystemStorage()
                        fs.save(classes_file_save_path, f)

                models = request.FILES.get("model")
                try:
                    os.makedirs('static/weights')
                except:
                    pass
                model_save_path = f"static/weights/audio_model.h5"
                try:
                    os.remove(model_save_path)
                except:
                    pass
                fs = FileSystemStorage()
                fs.save(model_save_path, models)

                res_status, class_res, confidences, all_classes = inference_audio_file('static/dataset', model_save_path, test_file_save_path)
                all_classes = list(all_classes.values())
                confidences = [float(a) for a in confidences]
                context["status"] = "success"
                context["Predicted Class"] = class_res
                context["Confidences"] = confidences
                context["All Classes"] = all_classes

            elif main_flag == 'text_classification':

                test_file = request.POST.get("input_text")

                classes_file = request.FILES.get("data_file")

                classes_file_save_path = f"static/train_text_classes.json"
                try:
                    os.remove(classes_file_save_path)
                except:
                    pass

                fs = FileSystemStorage()
                fs.save(classes_file_save_path, classes_file)

                models = request.FILES.get("model")
                try:
                    os.system("rm -r static/text_classification_model")
                except:
                    pass
                model_save_path = "static/text_classification_model.zip"
                try:
                    os.remove(model_save_path)
                except:
                    pass
                fs = FileSystemStorage()
                fs.save(model_save_path, models)

                # os.system(f"unzip {model_save_path}")
                unzip_file(model_save_path)

                res_status, class_res, classes = text_inference(test_file)
                conf = class_res["score"]
                rem_conf = (1 - conf) / (len(classes) - 1)

                all_classes = [class_res["label"]]
                all_confs = [class_res["score"]]

                for cl in classes:
                    if cl not in all_classes:
                        all_classes.append(cl)
                        all_confs.append(rem_conf)

                context["status"] = "success"
                context["Predicted Class"] = class_res["label"]
                context["Confidences"] = all_confs
                context["All Classes"] = all_classes
            else:
                context["status"] = "Failed"
                context["class"] = "Invalid Main Flag"
            return JsonResponse(context)
        except Exception as e:
            context = {
                "status": "failed",
                "error": str(e)
            }
            return JsonResponse(context)
    else:
        context = {
            "status": "failed",
            "error": "request type should be POST"
        }
        return JsonResponse(context)


def unzip_file(zip_file):
    try:
        # import zipfile module
        with ZipFile(zip_file, 'r') as f:
            # extract in current directory
            f.extractall('static/text_classification_model')
    except Exception as e:
        print(e)
