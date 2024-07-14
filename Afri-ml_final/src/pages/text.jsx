import { useEffect, useState, useRef } from "react";
import TopBannner from "@/components/TopBanner/TopBanner";
import Footer from "@/components/Footer/footer";
import TextMergingCard from "../components/TextMergingCard/textMergingCard";
import TrainingCard from "@/components/Cards/TrainingCard";
import TextPreviewCard from "@/components/Cards/TextPreviewCard";
import { coordinates, connectElements } from "@/lib/utils";
import Papa from "papaparse";
import { Upload, Pencil, MoreVertical } from "lucide-react";
import SmilingKid from "@/assets/smiling-kid.webp";
import textClass from "@/assets/textClass.json";
import textModel from "../assets/textData.zip";

const defaultData = {
  classTitle: "Class 2",
  classIcon: <Pencil size={16} />,
  toolIcon: <MoreVertical size={20} />,
  dataTool: "Text Editor",
  actionText: "Add Text",
  actionIcon: <> </>,
  sampleText: "Upload Document",
  sampleIcon: <Upload size={20} />,
};

const cardData = [
  {
    classTitle: "Class 1",
    classIcon: <Pencil size={16} />,
    toolIcon: <MoreVertical size={20} />,
    dataTool: "Text Editor",
    actionText: "Add Text",
    actionIcon: <> </>,
    sampleText: "Upload Document",
    sampleIcon: <Upload size={20} />,
  },
  defaultData,
];

const selectList = [
  {
    label: "Text",
    value: "text",
  },
  {
    label: "Documents",
    value: "documents",
  },
];

const Text = () => {
  const [connections2, setConnections2] = useState([]);
  const [classes, setClasses] = useState([...cardData]);
  const [checked, setChecked] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [success, setSuccess] = useState("");
  const [textSrcLengths, setTextSrcLengths] = useState([]);
  const [isPreviewEnabled, setIsPreviewEnabled] = useState(true);
  const [capturedText, setCapturedText] = useState("");
  const [confidence, setConfidence] = useState([]);
  const fileInputRef = useRef(null);

  const trainingHandler = async () => {
    setSuccess("");
    try {
      for (const classItem of classes) {
        if (
          !classItem.classTitle ||
          !(classItem.textSrc && classItem.textSrc.length)
        ) {
          setError(`Class Title or Text Source cannot be empty for classes`);
          console.log("error", error);
          return;
        }
        if (classItem.textSrc.length < 10) {
          setError("Text should be more than 10 files");
          return;
        }
      }
      setError("");
      setIsLoading(true);

      const uploadUrl = process.env.REACT_APP_UPLOAD_URL;
      const trainUrl = process.env.REACT_APP_PROCESS_URL;
      const responses = [];

      const allClassData = [];
      for (const classItem of classes) {
        const classData = classItem.textSrc.map((text) => ({
          text: text.content,
          label: classItem.classTitle,
        }));
        allClassData.push(...classData);
      }

      const csvData = Papa.unparse(allClassData);
      console.log("csvData", csvData);

      const blob = new Blob([csvData], { type: "text/csv" });
      const csvFile = new File([blob], "train_text.csv", { type: "text/csv" });

      const formData = new FormData();

      formData.append("main_flag", "text_classification");
      formData.append("images_list", csvFile);

      const response = await fetch(uploadUrl, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        console.log("result", result);
        responses.push(result.status === "success" ? "success" : result.status);
      } else {
        throw new Error(
          `Failed to upload data for class ${classItem.classTitle}`
        );
      }

      const trainFormData = new FormData();
      trainFormData.append("main_flag", "text_classification");
      trainFormData.append("flag", "train");

      const trainResponse = await fetch(trainUrl, {
        method: "POST",
        body: trainFormData,
      });
      if (trainResponse.ok) {
        const trainResult = await trainResponse.json();
        console.log("Train result", trainResult);
        responses.push(
          trainResult.status === "success" ? "success" : trainResult.status
        );
      } else {
        throw new Error(
          `Failed to upload data for class ${classItem.classTitle}`
        );
      }
      const isSuccess = responses.every((status) => status === "success");
      console.log("isSuccess", isSuccess);
      if (isSuccess) {
        setSuccess(<p className="text-xs">Model Trained Successfully</p>);
        setIsPreviewEnabled(true);
      } else {
        setSuccess(
          <p className="text-xs text-red-500">Something went wrong</p>
        );
      }
    } catch {
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };
  const handleAdvancedTest = async () => {
    const advancedTestUrl = process.env.REACT_APP_EXPORT_URL;
  
    const formData = new FormData();
    formData.append("main_flag", "text_classification");
  
    if (capturedText) {
      formData.append("input_text", capturedText);
    } else if (fileInputRef.current.files.length > 0) {
      formData.append("input_text", fileInputRef.current.files[0]);
    } else {
      console.log("No text file selected.");
      return;
    }
  
    const fetchModelFile = async (filePath, fileName) => {
      const response = await fetch(filePath);
      const blob = await response.blob();
      return new File([blob], fileName);
    };
  
    try {
      const modelFile = await fetchModelFile(textModel, "textData.zip");
      formData.append("model", modelFile);
  
      const response = await fetch("/src/assets/textClass.json");
      const json = await response.json();
      const blob = new Blob([JSON.stringify(json)], { type: "application/json" });
      formData.append("data_file", blob, "train_text_classes.json");
  
      const advancedResponse = await fetch(advancedTestUrl, {
        method: "POST",
        body: formData,
      });
  
      if (advancedResponse.ok) {
        const advancedResult = await advancedResponse.json();
        console.log("advancedResult", advancedResult);
        setConfidence(advancedResult?.Confidences);
      } else {
        console.log("Something went wrong with the advanced API call");
      }
    } catch (error) {
      console.error("Error during advanced API call:", error);
    }
  };
  useEffect(() => {
    const lengths = classes.map((classItem) =>
      classItem.textSrc ? classItem.textSrc.length : 0
    );
    setTextSrcLengths(lengths);
  }, [classes]);

  useEffect(() => {
    connectElements(setConnections2);
  }, [classes]);

  useEffect(() => {
    coordinates(connections2);

    const handleScroll = () => {
      coordinates(connections2);
    };

    const scrollElement = document.querySelector(".classes-container");

    if (scrollElement) {
      scrollElement.addEventListener("scroll", handleScroll);

      return () => {
        scrollElement.removeEventListener("scroll", handleScroll);
      };
    }
  }, [connections2]);

  return (
    <>
      <TopBannner title={"Texts"} pageLink={"Texts"} />
      <svg height="1000" width="1000" id="paths"></svg>
      <div className="container mx-auto px-4 ">
        <div className="mv-sequence my-10 relative">
          <div className="space-y-4">
            <div className="classes-container overflow-y-auto max-h-[800px] [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] flex flex-col gap-4">
              {classes.map((data, index) => (
                <TextMergingCard
                  key={index}
                  handleEditClassNames={(name) =>
                    setClasses(
                      classes.map((d, i) =>
                        i === index ? { ...d, classTitle: name } : d
                      )
                    )
                  }
                  handleDeleteClass={() =>
                    setClasses(classes.filter((_, i) => i !== index))
                  }
                  handleAddText={(newTextEntries) =>
                    setClasses(
                      classes.map((d, i) =>
                        i === index ? { ...d, textSrc: newTextEntries } : d
                      )
                    )
                  }
                  {...data}
                  textSrcLengths={textSrcLengths[index]}
                />
              ))}
            </div>
            <div
              className="relative w-[550px] h-[94px] rounded-[10px] border-[3px] border-dashed border-[#bdc1c6] flex justify-center items-center gap-2 cursor-pointer"
              onClick={() =>
                setClasses([
                  ...classes,
                  { ...defaultData, classTitle: `Class ${classes.length + 1}` },
                ])
              }
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="15"
                height="14"
                viewBox="0 0 15 14"
                fill="none"
              >
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M12.5161 0.25H1.85376C1.00838 0.25 0.330566 0.925 0.330566 1.75V12.25C0.330566 13.075 1.00838 13.75 1.85376 13.75H12.5161C13.3539 13.75 14.0393 13.075 14.0393 12.25V1.75C14.0393 0.925 13.3539 0.25 12.5161 0.25ZM12.5161 12.25H1.85376V1.75H12.5161V12.25ZM7.94654 10.75H6.42335V7.75H3.37695V6.25H6.42335V3.25H7.94654V6.25H10.9929V7.75H7.94654V10.75Z"
                  fill="#80868B"
                />
              </svg>
              <div className="font-normal text-[#5f6368] text-[16px] text-center tracking-[0] leading-[normal]">
                Add a class
              </div>
            </div>
          </div>
          <div className="relative top-[320px]">
            <TrainingCard
              checked={checked}
              setChecked={setChecked}
              classes={classes}
              trainingHandler={trainingHandler}
              error={error}
              isLoading={isLoading}
              success={success}
            />
          </div>
          <TextPreviewCard
            action={"Microphone"}
            predictionData={classes}
            selectList={selectList}
            checked={checked}
            IsPreviewEnabled={isPreviewEnabled}
            SmilingKid={SmilingKid}
            handleAdvancedTest={handleAdvancedTest}
            capturedText={capturedText}
            setCapturedText={setCapturedText}
            fileInputRef={fileInputRef}
            confidence={confidence}
          />
        </div>
        <Footer />
      </div>
    </>
  );
};

export default Text;
