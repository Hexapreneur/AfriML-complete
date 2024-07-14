import React, { useEffect, useState, useRef } from "react";
import TopBannner from "@/components/TopBanner/TopBanner";
import Footer from "@/components/Footer/footer";
import AudioMergingCard from "../components/AudioMergingCard/audioMergingCard";
import TrainingCard from "@/components/Cards/TrainingCard";
import AudioPreviewCard from "@/components/Cards/AudioPreviewCard";
import { coordinates, connectElements } from "@/lib/utils";
import { Mic, Upload, HelpCircle, Pencil } from "lucide-react";
import AudioMainmerge from "../components/AudioMergingCard/audioMainmerge";
import dancingKid from "@/assets/dancing-kid.gif";
import audioModel from "@/assets/audio.h5";
import audioClasses from "@/assets/audioClasses.txt";
import audioData from "@/assets/audioData.csv";

const defaultData = {
  classTitle: "Class 1",
  classIcon: <Pencil size={16} />,
  toolIcon: <Mic size={20} />,
  dataTool: "Microphone (10 samples minimum)",
  actionText: "Hold to record",
  actionIcon: <Mic size={20} />,
  sampleText: "Upload Audio",
  sampleIcon: <Upload size={20} />,
};

const cardData = [
  {
    classTitle: "Background Noise",
    classIcon: <HelpCircle size={20} />,
    toolIcon: <Mic size={20} />,
    dataTool: "Microphone (10sec minimum)",
    actionText: "Hold to record",
    actionIcon: <Mic size={20} />,
    sampleText: "Upload Audio",
    sampleIcon: <Upload size={20} />,
  },
  defaultData,
];

const selectList = [
  {
    label: "Microphone",
    value: "microphone",
  },
  {
    label: "Audio",
    value: "audio",
  },
];

const Audio = () => {
  console.log("audioData", audioData);
  console.log("audioClasses", audioClasses);
  console.log("audioModel", audioModel);
  const [connections2, setConnections2] = useState([]);
  const [classes, setClasses] = useState([...cardData]);
  console.log("classes", classes);
  const [checked, setChecked] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [success, setSuccess] = useState("");
  const [audioSrcLengths, setAudioSrcLengths] = useState([]);
  const [isPreviewEnabled, setIsPreviewEnabled] = useState(false);
  const [capturedAudio, setCapturedAudio] = useState(null);
  const fileInputRef = useRef(null);
  const [confidence, setConfidence] = useState([]);
  const [advancedResult, setAdvancedResult] = useState(null);

  const trainingHandler = async () => {
    setSuccess("");
    try {
      for (const classItem of classes) {
        if (
          !classItem.classTitle ||
          !(classItem.audioSrc && classItem.audioSrc.length)
        ) {
          setError(`Class title or Audio source cannot be empty for classes`);
          console.log("error", error);
          return;
        }
        if (classItem.audioSrc.length < 10) {
          setError("Audio should be more than 10");
          return;
        }
      }
      setError("");
      setIsLoading(true);
      const createUrl = process.env.REACT_APP_CREATE_URL;
      const uploadUrl = process.env.REACT_APP_UPLOAD_URL;
      const trainUrl = process.env.REACT_APP_PROCESS_URL;

      const responses = [];
      const classes1 = classes.map((item) => item.classTitle);
      console.log("classes1", classes1);
      let commaSeparatedString = classes1.join(",");
      console.log(commaSeparatedString);
      const formData = new FormData();
      formData.append("main_flag", "audio_classification");

      formData.append("classes", commaSeparatedString);
      const response = await fetch(createUrl, {
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

      for (const classItem of classes) {
        const formData = new FormData();
        formData.append("class_name", classItem.classTitle);

        classItem.audioSrc.forEach((file, index) => {
          formData.append("images_list", file);
        });

        formData.append("main_flag", "audio_classification");
        const uploadResponse = await fetch(uploadUrl, {
          method: "POST",
          body: formData,
        });

        if (uploadResponse.ok) {
          const result = await uploadResponse.json();
          console.log("result", result);
          responses.push(
            result.status === "success" ? "success" : result.status
          );
        } else {
          throw new Error(
            `Failed to upload data for class ${classItem.classTitle}`
          );
        }
      }
      const trainFormData = new FormData();
      trainFormData.append("main_flag", "audio_classification");
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
    formData.append("main_flag", "audio_classification");
  
    if (capturedAudio) {
      formData.append("test_file", capturedAudio);
    } else if (fileInputRef.current.files.length > 0) {
      formData.append("test_file", fileInputRef.current.files[0]);
    } else {
      console.log("No audio file selected.");
      return;
    }
  
    try {
      const fetchFile = async (filePath, fileName) => {
        const response = await fetch(filePath);
        const blob = await response.blob();
        return new File([blob], fileName);
      };
  
      const modelFile = await fetchFile(audioModel, "audio.h5");
      const classesFile = await fetchFile(audioClasses, "audioClasses.txt");
      const dataFile = await fetchFile(audioData, "audioData.csv");
  
      formData.append("model", modelFile);
      formData.append("data_file", classesFile);
      formData.append("data_file", dataFile);
  
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
      classItem.audioSrc ? classItem.audioSrc.length : 0
    );
    setAudioSrcLengths(lengths);
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
      <TopBannner title={"Audio"} pageLink={"Audio"} />
      <svg height="1000" width="1000" id="paths"></svg>
      <div className="container mx-auto px-4">
        <div className="mv-sequence my-10 relative">
          <div className="space-y-4">
            <div className="classes-container overflow-y-auto max-h-[800px] [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] flex flex-col gap-4">
              {classes.map((data, index) => (
                <AudioMergingCard
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
                  handleAddAudio={(newMediaFiles) =>
                    setClasses(
                      classes.map((d, i) =>
                        i === index ? { ...d, audioSrc: newMediaFiles } : d
                      )
                    )
                  }
                  {...data}
                  audioSrcLengths={audioSrcLengths[index]}
                />
              ))}
            </div>
            <div
              className="relative w-[550px] h-[94px] rounded-[10px] border-[3px] border-dashed border-[#bdc1c6] flex justify-center items-center gap-2 cursor-pointer"
              onClick={() =>
                setClasses([
                  ...classes,
                  { ...defaultData, classTitle: `Class ${classes.length}` },
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
              isLoading={isLoading}
              error={error}
              trainingHandler={trainingHandler}
              success={success}
            />
          </div>
          <AudioPreviewCard
            action={"Microphone"}
            predictionData={classes}
            selectList={selectList}
            checked={checked}
            isPreviewEnabled={isPreviewEnabled}
            dancingKid={dancingKid}
            handleAdvancedTest={handleAdvancedTest}
            capturedAudio={capturedAudio}
            setCapturedAudio={setCapturedAudio}
            fileInputRef={fileInputRef}
            confidence={confidence}
            advancedResult={advancedResult} //Pass it here
          />
        </div>
        <Footer />
      </div>
    </>
  );
};

export default Audio;
