import React from "react";
import MainAboutBanner from "../../assets/mainAboutBanner.png";
import { Button } from "@/components/ui/button";
import AboutChecboxes from "./AboutChecboxes";
import { Link } from "react-router-dom";
const mainAboutBanner = () => {
  const dataArray = [
    {
      descrpition: "Images",
    },
    {
      descrpition: "Audio",
    },
    {
      descrpition: "Text ",
    },
    {
      descrpition: "Poses",
    },
  ];

  return (
    <>
      <div className="mainAboutBanner w-11/12 mr-auto ml-auto flex py-20">
        {/* <div className="aboutbanner w-6/12">
          <img src={MainAboutBanner} alt="" className="w-90" />
        </div> */}
        <div id="image_block_01">
          <div
            className="image-box wow slideInLeft animated"
            data-wow-delay="0ms"
            data-wow-duration="1500ms"
            // style="isibility: visible; animation-duration: 1500ms; animation-delay: 0ms; animation-name: slideInLeft;"
          >
            <figure
              className="image js-tilt"
            >
              <img src={MainAboutBanner} alt="" />
              <div
                className="js-tilt-glare"
              >
                <div
                  className="js-tilt-glare-inner"
                ></div>
              </div>
            </figure>
          </div>
        </div>
        <div className="aboutbannerpic w-6/12 text-left">
          <span className="text-lg text-primary">ABOUT AFRI-ML</span>
          <br />
          <br />
          <h1 className="sm:text-4xl md:text-4xl text-2xl leading-tight font-bold text-left ">
            Machine Learning <br />
            for African Kids.
          </h1>{" "}
          <br />
          <span className="text-base text-customgrey">
            This is a special computer program for African kids that will help
            them
            <br /> teach their computers to recognize images, sound and texts. 
            Embedded is the ability to identify African images, sounds and
            <br />language because it is trained to understand
            African culture, nature
            <br />and accent. It accepts these inputs:
          </span>
          <br />
          <br />
          <div className="flex flex-col gap-4">
            {dataArray.map((Data, index) => (
              <AboutChecboxes key={index} checkData={Data} />
            ))}
          </div>
          <br />
          <Button
            variant="outline"
            className="rounded-full text-white border-primary px-10  bg-primary "
          >
           <Link to="/about" className="z-10">
              
            {" "}
            Read More{" "}
            
            </Link>
          </Button>
        </div>
      </div>
    </>
  );
};

export default mainAboutBanner;
