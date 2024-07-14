import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Mic, HelpCircle } from "lucide-react";
import Mainmergecard from "./Mainmerge";
import { useEffect, useRef, useState } from "react";
import { Check } from "lucide-react";
import { Trash2 } from "lucide-react";

const MergingCard = ({
  classTitle,
  classIcon,
  toolIcon,
  handleEditClassNames,
  handleDeleteClass,
  handleAddImages,
  isActive,
  onActivate,
  srcLengths,
  poseSrcLengths,
  activeCam,
  ...props
}) => {
  const [currentClass, setCurrentClass] = useState(classTitle);
  const [editMode, setEditMode] = useState(false);
  const inputRef = useRef(null);
  const [imgSrc, setImgSrc] = useState([]);

  console.log("currentClass", currentClass);
  console.log("imgSrc", imgSrc);

  useEffect(() => {
    if (editMode) {
      inputRef.current.focus();
    }
  }, [editMode]);

  useEffect(() => {
    setCurrentClass(classTitle);
  }, [classTitle]);
  useEffect(() => {
    handleAddImages(imgSrc);
  }, [imgSrc]);

  return (
    <>
      <div className="flex justify-center ">
        <Card className="w-[550px] rounded-3xl shadow-md relative mv-command1">
          <CardHeader className="gap-4 border-b-2 border-b-grey">
            <CardTitle className="text-start  flex relative items-center">
              {editMode ? (
                <input
                  type="text"
                  value={currentClass}
                  onChange={(e) => setCurrentClass(e.target.value)}
                  placeholder="Enter new class title"
                  className="border-0 outline-0 h-7 leading-loose max-w-[200px]"
                  size={currentClass.length}
                  ref={inputRef}
                />
              ) : (
                <span className="leading-[28px] tracking-normal max-w-[200px] overflow-hidden whitespace-nowrap text-ellipsis">
                  {currentClass}
                </span>
              )}
              <Button
                variant="outline"
                className="border-0 h-2 p-3 text-customgrey"
                onClick={() => {
                  if (currentClass === "Background Noise") {
                    return;
                  }
                  if (!editMode) setEditMode(!editMode);
                  else {
                    handleEditClassNames(currentClass);
                    setEditMode(!editMode);
                  }
                }}
              >
                {!editMode ? classIcon : <Check size={20} />}
              </Button>
              {/* {currentClass !== "Background Noise" && (
                <Button
                  variant="outline"
                  className=" border-0 h-2 p-3 text-customgrey"
                  onClick={handleDeleteClass}
                >
                  <Trash2 size={20} />
                </Button>
              )} */}
              {currentClass !== "Background Noise" && (
                <Button
                  variant="outline"
                  className=" border-0 h-2 p-3 text-customgrey absolute right-0 "
                  onClick={handleDeleteClass}
                >
                  <Trash2 size={20} />
                </Button>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent></CardContent>
          <Mainmergecard
            {...props}
            imgSrcs={imgSrc}
            setImgSrcs={setImgSrc}
            isActive={isActive}
            onActivate={onActivate}
            srcLengths={srcLengths}
            poseSrcLengths={poseSrcLengths}
            activeCam={activeCam}
          />
        </Card>
      </div>
    </>
  );
};

export default MergingCard;
