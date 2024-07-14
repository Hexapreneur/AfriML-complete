import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronDown, Loader2, Upload } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
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
import { useState, useRef } from "react";
import Papa from "papaparse";
/*import quoteImage from "@/assets/Quote.png";


const phrases = {
  YORUBA: "Yoruba Language Detected.",
  HAUSA: "Hausa Language Detected.",
  IGBO: "Igbo Language Detected.",
};

const getPhrase = (predictedClass) => {
  return phrases[predictedClass] || "Default phrase"; // Fallback in case of unmatched class
};*/


const TextPreviewCard = ({
  action,
  predictionData,
  selectList,
  checked,
  IsPreviewEnabled,
  SmilingKid,
  handleAdvancedTest,
  capturedText,
  setCapturedText,
  fileInputRef,
  confidence,
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [selectedOption, setSelectedOption] = useState("");
  const [uploadFile, setUploadFile] = useState(null);
  const [fileContent, setFileContent] = useState("");
  const [csvFile, setCsvFile] = useState(null);
  const [resultClass, setResultClass] = useState([]);
  const [predictions, setPredictions] = useState({
    predictedClass: "",
    allClasses: [],
    confidences: [],
  });

  const handleSelect = (value) => {
    setSelectedOption(value);
    setCapturedText("");
    setUploadFile(null);
    setFileContent("");
    setCsvFile(null);
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        setFileContent(content);
        setCsvFile(file);
      };
      reader.readAsText(file);
    }
  };

  const handleUploadButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleTestButtonClick = async () => {
    const testUrl = process.env.REACT_APP_PROCESS_URL;
    const formData = new FormData();

    setIsLoading(true);
    formData.append("main_flag", "text_classification");
    formData.append("flag", "test");

    if (selectedOption === "documents" && csvFile) {
      formData.append("text", csvFile);
    } else if (selectedOption === "text" && capturedText) {
      formData.append("text", capturedText);
    } else {
      console.log("No file or text input provided.");
      setIsLoading(false);
      return;
    }

    try {
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

        if (checked) {
          handleAdvancedTest();
        }
      } else {
        console.log("Something Went Wrong");
      }
    } catch (error) {
      console.log(`An error occurred: ${error.message}`);
    } finally {
      setIsLoading(false);
    }

    setCapturedText("");
  };

  const exportHandler = async () => {
    setIsLoading(true);
    const exportUrl = process.env.REACT_APP_PROCESS_URL;

    const formData = new FormData();
    formData.append("main_flag", "text_classification");
    formData.append("flag", "export");

    try {
      const response = await fetch(exportUrl, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        console.log("result", result);

        const downloadFile = (fileUrl) => {
          const filename = fileUrl.substring(fileUrl.lastIndexOf("/") + 1);
          console.log("Final download URL:", fileUrl);
          const link = document.createElement("a");
          link.href = fileUrl;
          link.setAttribute("download", filename);
          link.setAttribute("target", "_blank");

          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        };

        if (result.Model) {
          downloadFile(result.Model);
        } else {
          console.log("Model URL not found in the result.");
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
  const isAnyConfidenceAbove90 = confidence?.some((conf) => conf > 0.9);

  //const phrase = getPhrase(predictions.predictedClass);

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
                    <SelectValue placeholder="Select an option" />
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
                <div className="flex flex-col justify-center align-center relative ">
                  {selectedOption === "text" && (
                    <textarea
                      className="w-full h-60 p-2 border rounded"
                      placeholder="Enter your text here..."
                      value={capturedText}
                      onChange={(e) => setCapturedText(e.target.value)}
                    />
                  )}
                  {selectedOption === "documents" && !uploadFile && (
                    <div className="w-full mt-2 h-full bg-white flex justify-center relative">
                      <input
                        type="file"
                        ref={fileInputRef}
                        style={{ display: "none" }}
                        accept=".txt"
                        onChange={handleFileChange}
                      />
                      <div className="flex justify-center relative">
                        <Button
                          className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1"
                          onClick={handleUploadButtonClick}
                        >
                          <Upload size={20} /> Upload CSV File
                        </Button>
                      </div>
                    </div>
                  )}
                  {selectedOption === "documents" && uploadFile && (
                    <>
                      <div className="flex flex-col justify-center align-center relative w-full">
                        <div className="p-4 overflow-auto w-full h-60 border rounded bg-white">
                          {fileContent}
                        </div>
                      </div>
                      <div className="flex justify-center">
                        <Button
                          className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1"
                          onClick={handleTestButtonClick}
                        >
                          {isLoading ? (
                            <Loader2 className="m-auto h-4 w-4 animate-spin " />
                          ) : (
                            "Test"
                          )}
                        </Button>
                      </div>
                    </>
                  )}
                  {selectedOption === "text" && (
                    <div className="flex justify-center">
                      <Button
                        className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1"
                        onClick={handleTestButtonClick}
                      >
                        {isLoading ? (
                          <Loader2 className="m-auto h-4 w-4 animate-spin " />
                        ) : (
                          "Test"
                        )}
                      </Button>
                    </div>
                  )}
                </div>
              </div>
              {!IsPreviewEnabled && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-50 z-10">
                  <span className="text-black text-lg">
                    Please train a model to enable preview.
                  </span>
                </div>
              )}
            </CardContent>
            <CardContent className="border-b border-b-grey pb-0">
              <div className="w-full h-20 flex items-center justify-between">
                <CardTitle className="flex items-center text-lg">
                  Prediction
                </CardTitle>
                {checked &&
                  isAnyConfidenceAbove90 && (
                      <img src={SmilingKid} alt="" className="w-20 h-20" />
                  )}
              </div>
            </CardContent>
            <CardFooter className="text-center flex justify-center items-center flex-col pt-6 gap-4 max-h-80">
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

export default TextPreviewCard;
