from django.shortcuts import render
from .ml_model import predict_crop, CROP_INFO


def index(request):
    return render(request, "crops/index.html")


def about(request):
    return render(request, "crops/about.html")


SOIL_TYPES = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
CROP_TYPES = [
    "Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy",
    "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts",
]

def predict(request):
    if request.method == "POST":
     
        raw_data = {
            'N': request.POST.get('N', '').strip(),
            'P': request.POST.get('P', '').strip(),
            'K': request.POST.get('K', '').strip(),
            'temperature': request.POST.get('temperature', '').strip(),
            'humidity': request.POST.get('humidity', '').strip(),
            'ph': request.POST.get('ph', '').strip(),
            'rainfall': request.POST.get('rainfall', '').strip(),
        }

        errors = {}
        validated_data = {}
        
        for key, value in raw_data.items():
            if not value:
                errors[key] = "This field is required."
            else:
                try:
                    validated_data[key] = float(value)
                except ValueError:
                    errors[key] = "Enter a valid number."

        if errors:
            return render(request, "crops/index.html", {
                "errors": errors,
                "post_data": raw_data
            })

        result = predict_crop(validated_data)
      
        if result.get("success"):
            context = {
                "result": result,
                "post_data": raw_data,          
                "main_crop_info": CROP_INFO.get(result["prediction"], {})
            }
            return render(request, "crops/result.html", context)
        else:
            
            return render(request, "crops/index.html", {
                "errors": {"general": result.get("error", "Prediction failed. Please try again.")},
                "post_data": raw_data
            })

    return render(request, "crops/index.html")