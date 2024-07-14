import React from "react";
import ProjectCard from "@/components/Cards/ProjectCard";
import project1 from "../../assets/project-1.jpg";
import project2 from "../../assets/project-2.jpg";
import project3 from "../../assets/project-3.jpg";
import project4 from "../../assets/project-4.jpg";
import project5 from "../../assets/project-5.jpg";
import project6 from "../../assets/project-6.jpg";
import bg from "@/assets/bg-3.png";

const Projects = () => {
  const cardDataArray = [
    {
      title: "Speech To Text",
      imgSrc: project1,
      description: "Scratch",
      url: "/project",
    },
    {
      title: "Accent Detection",
      imgSrc: project2,
      description: "Python",
      url: "/project",
    },
    {
      title: "Question Answering",
      imgSrc: project3,
      description: "Scratch",
      url: "/project",
    },
    {
      title: "Toxicity",
      imgSrc: project4,
      description: "Python",
      url: "/project",
    },
    {
      title: "Statistical Evaluation",
      imgSrc: project5,
      description: "Scratch",
      url: "/project",
    },
    {
      title: "Image To Text",
      imgSrc: project6,
      description: "Scratch",
      url: "/project",
    },
  ];

  return (
    <div className="container mx-auto px-4">
      <div className="innermain flex w-3/4 mr-auto ml-auto  mb-32 pt-10 ">
        <div className="flex flex-col w-full justify-center items-center relative">
          <div className="flex flex-col w-full items-start gap-4 relative ">
            <div className="flex flex-col w-full items-center gap-2 relative ">
              <div className="relative w-fit font-medium text-primary text-4 text-center tracking-[3.00px] whitespace-nowrap uppercase">
                Recent Projects
              </div>
              <div className="relative w-fit font-bold text-darkgrey text-4xl text-center leading-tight whitespace-nowrap">
                Check Some of Our Recent Work.
              </div>
            </div>
          </div>
          <div className="flex align-middle items-center justify-center pt-10 ">
          <div className="grid sm:grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-3 gap-8">
              {cardDataArray.map((cardData, index) => (
                <a key={index} href={cardData.url} target="_blank" rel="noopener noreferrer">
                  <ProjectCard key={index} cardData={cardData} />
                </a>  
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Projects;
