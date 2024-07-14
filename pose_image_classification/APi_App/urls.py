from django.urls import path
from . import views, export

urlpatterns = [
    path("process/", views.index, name="index"),
    path("create_classes_api/", views.create_classes_api, name="create_classes_api"),
    path("upload_data_api/", views.upload_data_api, name="upload_data_api"),
    path("get_classes_list/", views.get_classes_list, name="get_classes_list"),
    path("export/", export.export_function, name="export"),
]
