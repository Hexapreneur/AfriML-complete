import React, { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Mic } from "lucide-react";
import { Upload } from "lucide-react";
import Webcam from "react-webcam";

const Mainmerge = ({
  dataTool,
  actionText,
  actionIcon,
  sampleText,
  sampleIcon,
  imgSrcs,
  setImgSrcs,
  isActive,
  onActivate,
  srcLengths,
  poseSrcLengths,
  activeCam,
}) => {
  const fileInputRef = useRef(null);
  const webcamRef = useRef(null);

  const handleUploadButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  const handleFileChange = (e) => {
    Array.from(e.target.files).forEach((selectedFile) => {
      setImgSrcs((prevImgSrcs) => [...prevImgSrcs, selectedFile]);
    });
    e.target.value = null;
  };

  const capture = React.useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot({
        width: 60,
        height: 50,
      });
      fetch(imageSrc)
        .then((res) => res.blob())
        .then((blob) => {
          const file = new File([blob], "webcam-image.jpg", {
            type: "image/jpeg",
          });
          setImgSrcs((prevImgSrcs) => [...prevImgSrcs, file]);
        });
    }
  }, [webcamRef]);
  const videoConstraints = {
    width: { min: 280 },
    height: { min: 180 },
    aspectRatio: 1,
  };

  useEffect(() => {
    return () => {
      imgSrcs.forEach((file) => URL.revokeObjectURL(file));
    };
  }, [imgSrcs]);

  return (
    <>
      <div className="w-full flex -mt-6">
        <div className="main1 w-3/5 h-80 rounded-bl-3xl bg-[#bdc1c633] text-start flex flex-col items-start p-4 pt-2">
          <div className="flex gap-1">
            <span className="flex text-sm">Images should not cross (1000)</span>
          </div>
          <div className="w-full mt-2 h-full flex justify-center relative">
            {isActive  && activeCam ? (
              <>
                <Webcam
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  width={280}
                  height={340}
                />
                <Button
                  className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1"
                  onClick={capture}
                >
                  {actionIcon} {actionText}
                </Button>
              </>
            ) : (
              <div className="w-full h-full flex justify-center items-center">
                <Button
                  className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1"
                  onClick={onActivate}
                >
                  Show Webcam
                </Button>
              </div>
            )}
          </div>
        </div>
        <div className="w-2/5 h-80 bg-white rounded-br-3xl text-start flex flex-col items-start p-4 pt-2">
          <span className="text-sm">{srcLengths}-samples uploaded</span>
          <div className="flex  gap-2 flex-wrap overflow-y-auto h-[700px] [scrollbar-width:none] pt-3 w-full">
            {imgSrcs.map((file, index) => (
              <img
                key={index}
                src={URL.createObjectURL(file)}
                alt={`Uploaded ${index}`}
                style={{ width: "50px", height: "50px" }}
              />
            ))}
          </div>
          <div className="w-full mt-2 h-full bg-white flex justify-center relative ">
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: "none" }}
              multiple
              accept="image/jpeg, image/png"
              onChange={handleFileChange}
            />
            <Button
              className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1"
              onClick={handleUploadButtonClick}
            >
              {sampleIcon} {sampleText}
            </Button>
          </div>
        </div>
      </div>
    </>
  );
};

export default Mainmerge;
