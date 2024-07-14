/*import React from "react";
import TutorialCard from "@/components/Cards/TutorialCard";
import tutotialImg1 from "@/assets/tutorial-1.png";
import tutotialImg2 from "@/assets/tutorial-2.png";
import tutotialImg3 from "@/assets/tutorial-3.png";
import bg from "@/assets/rsz_logo_copy.png";
import link from "fs";

const Tutorials = () => {
  const cardDataArray = [
    {
      title: "Guide on how to use to Afri-ML maximally.",
      imgSrc: tutotialImg1,
      author: "David O.",
      commentscount: 3,
      date: "Dec 23",
    },
    {
      title: "What Is Machine Learning?",
      imgSrc: tutotialImg2,
      author: "Simplilearn",
      commentscount: 5,
      date: "Jan 1",
    },
    {
      title: "What is Artificial Intelligence?",
      imgSrc: tutotialImg3,
      author: "Simplilearn",
      commentscount: 4,
      date: "Jan 10",
    },
  ];

  return (
    <div className="relative">
      <img
        className="absolute top-28 left-0 w-[200px] h-[200px] opacity-10"
        alt="Group"
        src={bg}
      />
      <div className="container mx-auto px-4">
        <div className="innermain flex w-3/4 mr-auto ml-auto  mb-32 pt-10 ">
          <div className="flex flex-col w-full justify-center items-center relative">
            <div className="flex flex-col w-full items-start gap-4 relative ">
              <div className="flex flex-col w-full items-center gap-2 relative ">
                <div className="relative w-fit  font-medium text-primary text-4 text-center tracking-[3.00px] whitespace-nowrap uppercase">
                  Tutorials
                </div>
                <div className="relative w-fit font-bold text-darkgrey text-4xl text-center leading-tight whitespace-nowrap">
                  Check Our Tutorials
                </div>
              </div>
            </div>
            <div className="flex align-middle items-center justify-center pt-10 ">
              <div className="grid sm:grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-3 gap-8">
                {cardDataArray.map((cardData, index) => (
                  <TutorialCard key={index} cardData={cardData} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Tutorials;
*/




import React from "react";
import TutorialCard from "@/components/Cards/TutorialCard";
import tutotialImg1 from "@/assets/tutorial-1.png";
import tutotialImg2 from "@/assets/tutorial-2.png";
import tutotialImg3 from "@/assets/tutorial-3.png";
import bg from "@/assets/rsz_logo_copy.png";

const Tutorials = () => {
  const cardDataArray = [
    {
      title: "Guide on how to use to Afri-ML maximally.",
      imgSrc: tutotialImg1,
      author: "David O.",
      commentscount: 3,
      date: "Dec 23",
      url: "https://example.com/tutorial-1",
      description: "Afri-ML is a web-based platform that teaches ML concepts with African cultural orientation."
    },
    {
      title: "What Is Machine Learning?",
      imgSrc: tutotialImg2,
      author: "Simplilearn",
      commentscount: 2116,
      date: "Jan 1",
      url: "https://www.youtube.com/watch?v=ukzFI9rgwfU&t=21s",
      description: "Machine learning is the general term for when computers learn from data."
    },
    {
      title: "What is Artificial Intelligence?",
      imgSrc: tutotialImg3,
      author: "Simplilearn",
      commentscount: 170,
      date: "Jun 10",
      url: "https://youtu.be/uMzUB89uSxU?si=Gw1ByL81MYVtHKqy",
      description: "Artificial Intelligence is making computer-based machines think and act as humans."
    },
  ];

  return (
    <div className="relative">
      <img
        className="absolute top-28 left-0 w-[200px] h-[200px] opacity-10"
        alt="Group"
        src={bg}
      />
      <div className="container mx-auto px-4">
        <div className="innermain flex w-3/4 mr-auto ml-auto mb-32 pt-10">
          <div className="flex flex-col w-full justify-center items-center relative">
            <div className="flex flex-col w-full items-start gap-4 relative">
              <div className="flex flex-col w-full items-center gap-2 relative">
                <div className="relative w-fit font-medium text-primary text-4 text-center tracking-[3.00px] whitespace-nowrap uppercase">
                  Tutorials
                </div>
                <div className="relative w-fit font-bold text-darkgrey text-4xl text-center leading-tight whitespace-nowrap">
                  Check Out Tutorials
                </div>
              </div>
            </div>
            <div className="flex align-middle items-center justify-center pt-10">
              <div className="grid sm:grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-3 gap-8">
                {cardDataArray.map((cardData, index) => (
                  <a key={index} href={cardData.url} target="_blank" rel="noopener noreferrer">
                    <TutorialCard cardData={cardData} />
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Tutorials;
