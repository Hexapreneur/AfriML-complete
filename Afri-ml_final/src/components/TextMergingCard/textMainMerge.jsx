import React, { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Upload } from "lucide-react";

const TextMainmerge = ({
  dataTool,
  actionText,
  actionIcon,
  sampleText,
  sampleIcon,
  mediaFiles,
  setMediaFiles,
  textSrcLengths,
}) => {
  const fileInputRef = useRef(null);
  const [textAreaContent, setTextAreaContent] = useState("");

  const handleUploadButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = async (e) => {
    const files = Array.from(e.target.files);
    e.target.value = null;

    const contents = await Promise.all(files.map(readFileContent));
    const parsedContents = contents.flatMap(({ content }) =>
      parseContentToLines(content)
    );
    setMediaFiles((prevContents) => [...prevContents, ...parsedContents]);
  };

  const readFileContent = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => resolve({ content: event.target.result });
      reader.onerror = reject;
      reader.readAsText(file);
    });
  };

  const parseContentToLines = (content) => {
    return content
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line !== "")
      .map((line) => ({ content: line }));
  };

  const handleConvertButtonClick = () => {
    if (textAreaContent.trim() === "") return;

    const lines = textAreaContent
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line !== "");

    const parsedContents = lines.map((line) => ({ content: line }));

    setMediaFiles((prevContents) => [...prevContents, ...parsedContents]);
    setTextAreaContent("");
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
          <textarea
            name="text"
            id="text"
            className="h-full"
            value={textAreaContent}
            onChange={(e) => setTextAreaContent(e.target.value)}
          ></textarea>
          <Button
            className="absolute bottom-2 bg-blue-100 text-blue-600 hover:bg-blue-200 gap-1"
            onClick={handleConvertButtonClick}
          >
            {actionIcon} {actionText}
          </Button>
        </div>
      </div>
      <div className="w-2/5 h-80 bg-white rounded-br-3xl text-start flex flex-col items-start p-4 pt-2">
        <span className="text-sm">{textSrcLengths}-samples uploaded </span>
        <div className="flex gap-2 flex-wrap overflow-y-auto h-[700px] [scrollbar-width:none] pt-3 w-full">
          {mediaFiles?.map((file, index) => (
            <div key={index} className="w-full p-2 border rounded">
              <strong>{file.name}</strong>
              <pre
                style={{
                  maxWidth: "100%",
                  overflowWrap: "break-word",
                  wordWrap: "break-word",
                  whiteSpace: "pre-wrap",
                }}
              >
                {file.content}
              </pre>
            </div>
          ))}
        </div>
        <div className="w-full mt-2 h-full bg-white flex justify-center relative">
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: "none" }}
            multiple
            accept=".txt"
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

export default TextMainmerge;
