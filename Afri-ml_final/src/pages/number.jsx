import { useEffect, useState } from "react";
import TopBannner from "@/components/TopBanner/TopBanner";
import Footer from "@/components/Footer/footer";
import MergingCard from "../components/MergingCard/MergingCard";
import TrainingCard from "@/components/Cards/TrainingCard";
import PreviewCard from "@/components/Cards/PreviewCard";
import { coordinates, connectElements } from "@/lib/utils";
import {
  Mic,
  Upload,
  HelpCircle,
  Pencil,
  MoreVertical,
  Video,
} from "lucide-react";

const defaultData = {
  classTitle: "Class 2",
  classIcon: <Pencil size={16} />,
  toolIcon: <MoreVertical size={20} />,
  dataTool: "Text Editor",
  actionText: "Paste Clipboard",
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
    actionText: "Paste Clipboard",
    actionIcon: <> </>,
    sampleText: "Upload Document",
    sampleIcon: <Upload size={20} />,
  },
  defaultData,
];

const selectList = [
  {
    label: "Number",
    value: "number",
  },
  {
    label: "Documents",
    value: "documents",
  },
];

const Number = () => {
  const [connections2, setConnections2] = useState([]);
  const [classes, setClasses] = useState([...cardData]);
  const [checked, setChecked] = useState(false);

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
      <TopBannner title={"Numbers"} pageLink={"Numbers"} />
      <svg height="1000" width="1000" id="paths"></svg>
      <div className="container mx-auto px-4 ">
        <div className="mv-sequence my-10 relative">
          <div className="space-y-4">
            <div className="classes-container overflow-y-auto max-h-[800px] [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] flex flex-col gap-4">
              {classes.map((data, index) => (
                <MergingCard
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
                  {...data}
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
            <TrainingCard checked={checked} setChecked={setChecked} />
          </div>
          <PreviewCard
            action={"Microphone"}
            predictionData={classes}
            selectList={selectList}
            checked={checked}
          />
        </div>
        <Footer />
      </div>
    </>
  );
};

export default Number;
