import { useEffect, useState, useRef } from "react";
import TopBannner from "@/components/TopBanner/TopBanner";
import Footer from "@/components/Footer/footer";
import PreviewProjectCard from "@/components/Cards/PreviewProjectCard";
import jumpingKid from "@/assets/jumping-kid.gif";
import { Mic, Upload, Pencil, Camera, Video } from "lucide-react";



const cardData = [
  {
    classTitle: "Predicted Class",
    //classIcon: <Pencil size={16} />,
    //toolIcon: <Camera size={20} />,
    dataTool: "Webcam",
    actionText: "Click To Capture",
    actionIcon: <Video size={20} />,
    sampleText: "Upload Images",
    sampleIcon: <Upload size={20} />,
  },

];

const selectList = [
  {
    label: "Webcam",
    value: "webcam",
  },
  {
    label: "Image",
    value: "image",
  },
  {
    label: "Livecam",
    value: "livecam",
  },
];

const Image = () => {
  const [classes, setClasses] = useState([...cardData]);
  const [checked, setChecked] = useState(false);
  const fileInputRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);
  const [success, setSuccess] = useState("");
  const [isPreviewEnabled, setIsPreviewEnabled] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [confidence, setConfidence] = useState(0);

  const handleAdvancedTest = async () => {
    const advancedTestUrl = "http://127.0.0.1:8000/export/";

    const formData = new FormData();
    formData.append("main_flag", "image_classification");

    if (capturedImage) {
      formData.append("test_file", capturedImage);
    } else if (fileInputRef.current.files.length > 0) {
      formData.append("test_file", fileInputRef.current.files[0]);
    } else {
      console.log("No image or file selected.");
      return;
    }

    try {
      const advancedResponse = await fetch(advancedTestUrl, {
        method: "POST",
        body: formData,
      });

      if (advancedResponse.ok) {
        const advancedResult = await advancedResponse.json();
        console.log("advancedResult333", advancedResult);
        setConfidence(advancedResult?.Confidence);
      } else {
        console.log("Something went wrong with the advanced API call");
      }
    } catch (error) {
      console.error("Error during advanced API call:", error);
    }
  };

  const exportHandler = async () => {
    setIsLoading(true);
    const exportUrl = "http://127.0.0.1:8000/process/";

    const formData = new FormData();
    formData.append("main_flag", "image_classification");
    formData.append("flag", "export");

    try {
      const response = await fetch(exportUrl, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        console.log("result", result);

        if (result.Model) {
          const modelUrl = result.Model;

          const filename = modelUrl.substring(modelUrl.lastIndexOf("/") + 1);

          const link = document.createElement("a");
          link.href = modelUrl;
          link.download = filename;

          link.click();
        } else {
          console.log("Model URL not found in the result.");
        }
      } else {
        console.log(`Failed to fetch url `);
      }
    } catch (error) {
      console.log("Error during export:", error);
    }
    setIsLoading(false);
  };

  return (
    <>
      <TopBannner title={"Image"} pageLink={"Image"} />
      <div className="container mx-auto px-4 ">
        <PreviewProjectCard
          action={"Microphone"}
          predictionData={classes}
          selectList={selectList}
          checked={checked}
          isPreviewEnabled={isPreviewEnabled}
          jumpingKid={jumpingKid}
          handleAdvancedTest={handleAdvancedTest}
          capturedImage={capturedImage}
          setCapturedImage={setCapturedImage}
          fileInputRef={fileInputRef}
          exportHandler={exportHandler}
          classification="image_classification"
          confidence={confidence}
        />
      </div>
      <Footer />
    </>
  );
};

export default Image;
