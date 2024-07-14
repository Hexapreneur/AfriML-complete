import React from "react";
import mainAboutbannervector from "../../assets/mainAboutbannervector.png";

const AboutChecboxes = ({ checkData }) => {
  const { descrpition } = checkData;
  return (
    <>
      <div className="checkboxes text-left">
        <img src={mainAboutbannervector} alt="" className="inline-flex w-6" />{" "}
        <span className="text-sm text-customgrey ml-3">{descrpition}</span>
      </div>{" "}
    </>
  );
};

export default AboutChecboxes;
