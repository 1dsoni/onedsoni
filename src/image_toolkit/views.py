from django.shortcuts import render

from django.shortcuts import get_object_or_404
from django.http import HttpResponse, JsonResponse
from .apps import IMAGE_HELPER, OPENCV_HELPER, DLIB_HELPER
from django.views.decorators.csrf import csrf_exempt

# Create your views here.


@csrf_exempt
def image_toolkit_view( request):
        html_page = 'image_toolkit/image.html'
        context = {'success':False}
        if request.method == 'GET':
            return render( request, html_page,
                            context = context)

        elif request.method == 'POST':
            err = 'ERRORS'
            image = request.FILES.get("image", None)
            url = request.POST.get('url', None)
            json = request.POST.get('json', None)

            if image is not None:
                image_to_check = request.FILES.get("image").read()
                try:
                    img_cv_obj = OPENCV_HELPER( image=image_to_check )
                except Exception as e:
                    err += str( e) + '__\n'
                try :
                    img_dlib_obj = DLIB_HELPER( image=image_to_check)
                except Exception as e:
                    err += str(e) + '__\n'

            elif url is not None:
                try:
                    img_cv_obj = OPENCV_HELPER( url=url )
                except Exception as e:
                    err += str( e) + '__\n'
                try :
                    img_dlib_obj = DLIB_HELPER( url=url)
                except Exception as e:
                    err += str(e) + '__\n'
            if err != 'ERRORS':
                context = {'success':False, 'error':err}
                return JsonResponse( context )

                # return render( request, html_page, context=context)
            opencv_results = { 'num_faces':img_cv_obj.num_faces,
                            'faces':str( img_cv_obj.facial_points),
                            'marked_image_loc':img_cv_obj.marked_image_loc,}

            dlib_results = { 'num_faces':img_dlib_obj.num_faces,
                        'faces':str( img_dlib_obj.facial_points),
                        'marked_image_loc':img_dlib_obj.marked_image_loc,}

            context = { 'original_image':url,
                        'opencv_results':opencv_results,
                        'dlib_results':dlib_results, 'errors':err}

            if json is not None:
                return JsonResponse( context )
            return render( request, html_page
                        ,context = context)
