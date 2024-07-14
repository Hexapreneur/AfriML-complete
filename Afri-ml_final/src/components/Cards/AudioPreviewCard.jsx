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
import { useState, useRef } from "react";
import { AudioRecorder } from "react-audio-voice-recorder";
/*import quoteImage from "@/assets/Quote.png";

const phrases = {
  Background_Noise: "Other Nigerian Accent detected.",
  YORUBA: "Nigerian Yoruba Accent Detected.",
  HAUSA: "Nigerian Hausa Accent Detected.",
  IGBO: "Nigerian Igbo Accent Detected.",
  EFIKIBIBIO: "Nigerian Efikibibio Accent Detected.",
  WESTERN: "Nigerians' Western Accent Detected.."
};*/


const AudioPreviewCard = ({
  action,
  predictionData,
  selectList,
  checked,
  isPreviewEnabled,
  dancingKid,
  handleAdvancedTest,
  capturedAudio,
  setCapturedAudio,
  fileInputRef,
  confidence, 
  advancedResult,
}) => {
  console.log("selectList", selectList);
  const [isLoading, setIsLoading] = useState(false);
  const [showAudioRecorder, setShowAudioRecorder] = useState(false);
  const [audioButton, setShowAudioButton] = useState(false);
  const [uploadAudio, setUploadAudio] = useState(null);
  const [resultClass, setResultClass] = useState([]);
  const [predictions, setPredictions] = useState({
    predictedClass: "",
    allClasses: [],
    confidences: [],
  });
  const [selectValue, setSelectValue] = useState([]);
  console.log("resultClass", resultClass);
  const audioRef = useRef(null);
  console.log("uploadAudio", uploadAudio);


  /*const getPhrase = (predictedClass, advancedResult) => {
    return phrases[predictedClass] || "Default phrase"; // Fallback in case of unmatched class
  };*/


  const handleUploadButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const refreshhandler = () => {
    setCapturedAudio(null);
    if (selectValue === "microphone") {
      setShowAudioRecorder(true);
    }
    if (selectValue === "audio") {
      setShowAudioButton(true);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadAudio(file);
      setCapturedAudio(file);
      setShowAudioButton(false);
    }
  };

  const handleSelect = (value) => {
    setSelectValue(value);
    console.log("value", value);
    if (value === "microphone") {
      setShowAudioRecorder(true);
    } else {
      setShowAudioRecorder(false);
    }
    if (value === "audio") {
      setShowAudioButton(true);
    } else {
      setShowAudioButton(false);
    }
    setCapturedAudio(null);
  };

  const handleTestButtonClick = async () => {
    const testUrl = "http://127.0.0.1:8000/process/";
    const formData = new FormData();

    setIsLoading(true);
    formData.append("main_flag", "audio_classification");
    formData.append("flag", "test");
    console.log("capturedAudio", capturedAudio);
    if (capturedAudio) {
      formData.append("image", capturedAudio);
    } else if (fileInputRef.current.files.length > 0) {
      formData.append("image", fileInputRef.current.files[0]);
    } else {
      console.log("No audio or file selected.");
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

  const exportHandler = async () => {
    setIsLoading(true);
    const exportUrl = "http://127.0.0.1:8000/process/";

    const formData = new FormData();
    formData.append("main_flag", "audio_classification");
    formData.append("flag", "export");

    try {
      const response = await fetch(exportUrl, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        console.log("result", result);

        const downloadFile = (url) => {
          const filename = url.substring(url.lastIndexOf("/") + 1);
          const link = document.createElement("a");
          link.href = url;
          link.setAttribute("download", filename);
          link.setAttribute("target", "_blank");

          document.body.appendChild(link);
          link.click();
          link.parentNode.removeChild(link);
        };

        if (result.Model) {
          downloadFile(result.Model);
        } else {
          console.log("Audio Model URL not found in the result.");
        }

        if (result.Data) {
          downloadFile(result.Data);
        } else {
          console.log("Data URL not found in the result.");
        }

        if (result.Classes) {
          downloadFile(result.Classes);
        } else {
          console.log("Classes URL not found in the result.");
        }
      } else {
        console.log(`Failed to fetch URL: ${response.statusText}`);
      }
    } catch (error) {
      console.log(`An error occurred: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  React.useEffect(() => {
    return () => {
      if (capturedAudio) {
        URL.revokeObjectURL(capturedAudio);
      }
    };
  }, [capturedAudio]);
  const isAnyConfidenceAbove90 = confidence?.some((conf) => conf > 0.9);

  //const phrase = getPhrase(predictions.predictedClass, advancedResult);

  return (
    <>
      <Dialog>
        <div className="flex justify-center ">
          <Card className="w-[470px] rounded-2xl !shadow-custom border-0 relative preview">
            <CardHeader className="py-3 flex flex-row border-b border-solid border-blue-500 border-opacity-30 flex justify-between">
              <CardTitle className="flex items-center  text-lg">
                Preview
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
                  "Export Models"
                )}
              </Button>
            </CardHeader>
            <CardContent className="text-center bg-[#bdc1c633] flex justify-between items-center pt-6">
              <div className="w-full border h-96 bg-white flex justify-center relative">
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
                {showAudioRecorder && !capturedAudio && (
                  <div className="flex flex-col justify-center align-center relative">
                    <AudioRecorder
                      ref={audioRef}
                      onRecordingComplete={(blob) => {
                        const file = new File([blob], "recorded_audio.wav", {
                          type: "audio/wav",
                        });
                        setCapturedAudio(file);
                        setShowAudioRecorder(false);
                      }}
                    />
                  </div>
                )}
                {capturedAudio && (
                  <div className="flex flex-col justify-center align-center relative">
                    <audio controls src={URL.createObjectURL(capturedAudio)} />
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
                      className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1 ml-16"
                      onClick={refreshhandler}
                    >
                      Refresh
                    </Button>
                  </div>
                )}

                {audioButton && !capturedAudio && (
                  <div className="w-full mt-2 h-full bg-white flex justify-center relative">
                    <input
                      type="file"
                      ref={fileInputRef}
                      style={{ display: "none" }}
                      accept="audio/wav"
                      onChange={handleFileChange}
                    />
                    <Button
                      className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1"
                      onClick={handleUploadButtonClick}
                    >
                      <Upload size={20} /> Upload Audio
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
            <CardContent className="border-b border-b-grey pb-0">
              <div className="w-full h-20 flex items-center justify-between">
                <CardTitle className="flex items-center text-lg">
                  Prediction
                </CardTitle>
                {checked &&
                  isAnyConfidenceAbove90 && (
                     <img src={dancingKid} alt="" className="w-20 h-20" />
                  )}
              </div>
            </CardContent>
            <CardFooter className="text-center flex justify-center items-center flex-col pt-6 gap-4  max-h-80">
              <div
                className="h-40 w-82 overflow-auto bg-white p-4 rounded-md gap-2 flex flex-col"
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
                      <span className="max-w-[115px] overflow-hidden whitespace-nowrap text-ellipsis">
                        {item.classTitle}
                      </span>
                    </CardTitle>
                    <div className="flex items-center gap-2">
                      <Progress
                        className="w-64 h-8 bg-slate-100 "
                        value={
                          predictions?.confidences?.length > index &&
                          predictions?.confidences[index] > 0
                            ? predictions?.confidences[index] * 100
                            : 0
                        }
                      />
                      <CardTitle className="w-12 flex items-center text-lg">
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
                {!isPreviewEnabled && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-50 z-10">
                    <span className="text-black text-lg">
                      Please train a model to enable preview.
                    </span>
                  </div>
                )}
              </div>
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

export default AudioPreviewCard;
