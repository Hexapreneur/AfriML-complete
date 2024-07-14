import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronDown, Loader2, Mic, Upload } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import React from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";

const PreviewCard = ({
  action,
  predictionData,
  selectList,
  checked,
  isPreviewEnabled,
  jumpingKid,
  handleAdvancedTest,
  setCapturedImage,
  capturedImage,
  fileInputRef,
  exportHandler,
  classification,
  confidence,
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [showWebcam, setShowWebcam] = useState(false);
  const [showLivecam, setShowLivecam] = useState(false);
  const [imgButton, setShowImgButton] = useState(false);
  const webcamRef = useRef(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [predictions, setPredictions] = useState({
    predictedClass: "",
    allClasses: [],
    confidences: [],
  });
  console.log("predictions", predictions);
  const [resultClass, setResultClass] = useState([]);
  const [selectValue, setSelectValue] = useState([]);

  const handleUploadButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const refreshhandler = () => {
    setCapturedImage(null);
    if (selectValue === "webcam") {
      setShowWebcam(true);
    }
    if (selectValue === "image") {
      setShowImgButton(true);
    }
    if (selectValue === "livecam") {
      setShowLivecam(true);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedImage(file);
      setCapturedImage(file);
      setShowImgButton(false);
    }
  };

  const handleSelect = (value) => {
    setSelectValue(value);
    console.log("value", value);
    if (value === "webcam") {
      setShowWebcam(true);
      setShowLivecam(false);
    } else {
      setShowWebcam(false);
    }
    if (value === "image") {
      setShowImgButton(true);
      setShowLivecam(false);
    } else {
      setShowImgButton(false);
    }
    if (value === "livecam") {
      setShowLivecam(true);
    } else {
      setShowLivecam(false);
    }
    setCapturedImage(null);
  };

  const capture = React.useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot({
      width: 420,
      height: 315,
    });

    const byteCharacters = atob(imageSrc.split(",")[1]);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: "image/jpeg" });

    const file = new File([blob], "captured_image.jpeg", {
      type: "image/jpeg",
    });

    setCapturedImage(file);
    setShowWebcam(false);
  }, [webcamRef]);

  const handleTestButtonClick = async () => {
    const testUrl = "http://127.0.0.1:8000/process/";
    const formData = new FormData();

    setIsLoading(true);
    formData.append("main_flag", classification);
    formData.append("flag", "test");
    if (capturedImage) {
      formData.append("image", capturedImage);
    } else if (fileInputRef.current.files.length > 0) {
      formData.append("image", fileInputRef.current.files[0]);
    } else {
      console.log("No image or file selected.");
      return;
    }
    const response = await fetch(testUrl, {
      method: "POST",
      body: formData,
    });
    if (response.ok) {
      const result = await response.json();
      console.log("result", result);
      setPredictions({
        predictedClass: result["Predicted Class"],
        allClasses: result["All Classes"],
        confidences: result["Confidences"],
      });
      setResultClass(result["Predicted Class"]);
    } else {
      console.log("Something Went Wrong");
    }
    if (checked) {
      handleAdvancedTest();
    }
    setIsLoading(false);
  };

  const callAPI = async () => {
    console.log("capturedImage", capturedImage);
    const testUrl = "http://127.0.0.1:8000/process/";
    const formData = new FormData();
    formData.append("main_flag", "image_classification");
    formData.append("flag", "test");
    formData.append("image", capturedImage);
    if (capturedImage && showLivecam) {
      const response = await fetch(testUrl, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setPredictions({
          predictedClass: result["Predicted Class"],
          allClasses: result["All Classes"],
          confidences: result["Confidences"],
        });
        setResultClass(result["Predicted Class"]);
      } else {
        console.log("Something went wrong");
      }
      if (checked) {
        handleAdvancedTest();
      }
    }
    setIsLoading(false);
  };
  const startContinuousTesting = () => {
    setShowLivecam(true);
    setIsLoading(true);

    const interval = setInterval(() => {
      capture();
    }, 2000);
    return () => clearInterval(interval);
  };

  useEffect(() => {
    callAPI();
  }, [capturedImage]);

  useEffect(() => {
    if (showLivecam) {
      startContinuousTesting();
    }
  }, [showLivecam]);

  const videoConstraints = {
    width: { min: 240 },
    height: { min: 100 },
    aspectRatio: 1.111,
  };

  React.useEffect(() => {
    return () => {
      if (capturedImage) {
        URL.revokeObjectURL(capturedImage);
      }
    };
  }, [capturedImage]);

  return (
    <>
      <Dialog>
        <div className="flex justify-center ">
          <Card className="w-[1500px] rounded-2xl !shadow-custom border-0 relative preview">
            <CardHeader className="py-3 flex flex-row border-b border-solid border-blue-500 border-opacity-30 flex justify-between">
              <CardTitle className="flex items-center  text-lg">
                Project Testing
              </CardTitle>
              <Button
                className="bg-blue-100 text-gray hover:bg-blue-200 gap-2 px-2"
                onClick={exportHandler}
                disabled={isLoading}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 14 14"
                  fill="none"
                >
                  <path
                    d="M11.9997 4.49992H13.6663V1.99992C13.6663 1.08325 12.9163 0.333252 11.9997 0.333252H1.99967C1.08301 0.333252 0.333008 1.08325 0.333008 1.99992V4.49992H1.99967V1.99992H11.9997V4.49992ZM6.16634 6.85825L4.00801 9.00825L2.83301 7.83325L6.99967 3.66659L11.1663 7.83325L9.99134 9.01659L7.83301 6.85825V13.6666H6.16634V6.85825Z"
                    fill="#5F6368"
                  />
                </svg>{" "}
                {isLoading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  "Import Models"
                )}
              </Button>
            </CardHeader>
            <CardContent className="text-center bg-[#bdc1c633] flex justify-between items-center pt-6">
              <div className="w-full border bg-white flex justify-center relative"  style={{ height: '40rem' }}>
                <Select onValueChange={handleSelect}>
                  <SelectTrigger className="w-auto text-xs  absolute top-2 right-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1 rounded-full h-6 py-0 px-3 m-0 z-10">
                    <SelectValue placeholder="Select a option" />
                  </SelectTrigger>

                  <SelectContent>
                    <SelectGroup>
                      {selectList.map((item, index) => (
                        <SelectItem
                          key={`${item.value}${index}`}
                          value={item.value}
                          className="text-xs"
                        >
                          {item.label}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
                {showWebcam && !capturedImage && (
                  <div className="flex flex-col justify-center align-center relative ">
                    <Webcam
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      videoConstraints={videoConstraints}
                    />
                    <div className="flex justify-center">
                      <Button
                        className="absolute bottom-2  bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1"
                        onClick={capture}
                      >
                        <Upload size={20} /> Click To Capture
                      </Button>
                    </div>
                  </div>
                )}
                {capturedImage && !showLivecam && (
                  <div className="flex flex-col justify-center align-center relative">
                    <img
                      src={URL.createObjectURL(capturedImage)}
                      alt="Captured"
                      style={{ width: "100%", height: "100%" }}
                    />
                    <div className="flex justify-center gap-10">
                      <Button
                        className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1 ml-20"
                        onClick={handleTestButtonClick}
                      >
                        {isLoading ? (
                          <Loader2 className="m-auto h-4 w-4 animate-spin " />
                        ) : (
                          "Test"
                        )}
                      </Button>
                    </div>
                    <Button
                      className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1 ml-24"
                      onClick={refreshhandler}
                    >
                      Refresh
                    </Button>
                  </div>
                )}

                {imgButton && !capturedImage && (
                  <div className="w-full mt-2 h-full bg-white flex justify-center relative">
                    <input
                      type="file"
                      ref={fileInputRef}
                      style={{ display: "none" }}
                      accept="image/jpeg, image/png"
                      onChange={handleFileChange}
                    />
                    <Button
                      className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1"
                      onClick={handleUploadButtonClick}
                    >
                      <Upload size={20} /> Upload Image
                    </Button>
                  </div>
                )}
                {showLivecam && (
                  <div className="flex flex-col justify-center align-center relative ">
                    <Webcam
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      videoConstraints={videoConstraints}
                    />
                  </div>
                )}
              </div>
            </CardContent>
            <CardContent className="border-b border-b-grey pb-0">
              <div className="w-full h-0 flex items-center justify-between">
                
                {checked &&
                  confidence >
                    90 % <img src={jumpingKid} alt="" className="w-20 h-20" />}
              </div>
            </CardContent>
            <CardFooter className="text-center flex justify-center items-center flex-col pt-6 gap-4  max-h-80">
              <div
                className="h-full w-full overflow-auto bg-white p-4 rounded-md gap-2 flex flex-col"
                style={{
                  overflowY: "auto",
                  overflowX: "auto",
                  scrollbarWidth: "none",
                }}
              >
                {predictionData.map((item, index) => (
                  <div
                    key={index}
                    className="w-full flex items-center justify-between relative gap-2"
                  >
                    <CardTitle className="flex items-center text-lg">
                      <span className="max-w-[1150px] overflow-hidden whitespace-nowrap text-ellipsis">
                        {item.classTitle}
                      </span>
                    </CardTitle>
                    <div className=" flex items-center gap-2">
                      <Progress
                        className="w-64 h-8 bg-slate-100 "
                        value={
                          predictions?.confidences?.length > index &&
                          predictions?.confidences[index] > 0
                            ? predictions?.confidences[index] * 100
                            : 0
                        }
                      />
                      <CardTitle className="w-full flex items-center text-lg">
                        {predictions?.confidences?.length > index &&
                        predictions?.confidences[index] > 0
                          ? `${(predictions?.confidences[index] * 100).toFixed(
                              2
                            )}%`
                          : "0%"}
                      </CardTitle>
                    </div>
                  </div>
                ))}
              </div>
              {!isPreviewEnabled && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-50 z-10">
                  <span className="text-black text-lg">
                    
                  </span>
                </div>
              )}
              <CardDescription className="text-customgrey text-sm">
                You must train a model on the left before you can preview it
                here.
              </CardDescription>
            </CardFooter>
          </Card>
        </div>
      </Dialog>
    </>
  );
};

export default PreviewCard;
