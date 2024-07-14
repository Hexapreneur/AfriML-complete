import React, { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Mic } from "lucide-react";
import { Upload } from "lucide-react";
import { AudioRecorder } from "react-audio-voice-recorder";

const AudioMainmerge = ({
  dataTool,
  actionText,
  actionIcon,
  sampleText,
  sampleIcon,
  mediaFiles,
  setMediaFiles,
  audioSrcLengths,
}) => {
  const fileInputRef = useRef(null);

  const handleUploadButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (e) => {
    Array.from(e.target.files).forEach((selectedFile) => {
      setMediaFiles((prevMediaFiles) => [...prevMediaFiles, selectedFile]);
    });
    e.target.value = null;
  };

  const addAudioElement = (blob) => {
    console.log(blob);
    const file = new File([blob], "audio-recording.wav", {
      type: "audio/wav",
    });
    setMediaFiles((prevMediaFiles) => [...prevMediaFiles, file]);
  };

  useEffect(() => {
    return () => {
      mediaFiles?.forEach((file) => URL.revokeObjectURL(file));
    };
  }, [mediaFiles]);

  return (
    <div className="w-full flex -mt-6">
      <div className="main1 w-3/5 h-80 rounded-bl-3xl bg-[#bdc1c633] text-start flex flex-col items-start p-4 pt-2">
        <span className="flex text-sm">{dataTool}</span>
        <div className="w-full mt-2 h-full flex justify-center relative items-center">
          <AudioRecorder onRecordingComplete={addAudioElement} />
        </div>
      </div>
      <div className="w-2/5 h-80 bg-white rounded-br-3xl text-start flex flex-col items-start p-4 pt-2">
        <span className="text-sm">{audioSrcLengths}-samples uploaded </span>
        <div className="flex gap-2 flex-wrap overflow-y-auto h-[700px] [scrollbar-width:none] pt-3 w-full">
          {mediaFiles?.map((file, index) =>
            file.type.startsWith("audio/") ? (
              <audio key={index} controls src={URL.createObjectURL(file)} />
            ) : (
              <img
                key={index}
                src={URL.createObjectURL(file)}
                alt={`Uploaded ${index}`}
                style={{ width: "50px", height: "50px" }}
              />
            )
          )}
        </div>
        <div className="w-full mt-2 h-full bg-white flex justify-center relative ">
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: "none" }}
            multiple
            accept="audio/wav"
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
  );
};

export default AudioMainmerge;
