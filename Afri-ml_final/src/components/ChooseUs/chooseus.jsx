import workingbannerpattern from "../../assets/workingbannerpattern.png";
import serviceBanner from "../../assets/service-1.png";
import workingbannervector from "../../assets/workingBannervector.png";

const ChooseUs = ({ cardData }) => {
  const {
    topTitle,
    title,
    titlePara,
    heading1,
    headingpara1,
    img1,
    heading2,
    headingpara2,
    img2,
    heading3,
    headingpara3,
    img3,
    style,
  } = cardData;
  return (
    <>
      <div className="w-full mx-auto bg-[#F7FAFF] py-20">
        <img
          src={workingbannerpattern}
          alt=""
          className=" absolute w-1/4 right-0"
        />
        <div className="woorkingmain  mx-auto   w-2/3 h-full flex z-50 items-center justify-between ">
          <div className="working w-2/4 text-left pr-10">
            <span className="text-md text-primary tracking-[3.00px]">
              {topTitle}
            </span>
            <br />
            <br />
            <h2 className="sm:text-3xl md:text-4xl text-2xl leading-tight font-extrabold text-left ">
              {title}
            </h2>
            <br />
            <span className="text-base text-customgrey">{titlePara}</span>
            <br />
            <br />
            <div className="flex flex-col relative gap-4 ml-14">
              <div className="justify-center flex flex-col p-3 text-[22px] ">
                <h1 className="ml-6 font-extrabold text-xl">{heading1}</h1>
                <h6 className="ml-6 mt-2 leading-6 text-base text-customgrey">
                  {headingpara1}
                </h6>
                <div className="absolute -left-16">
                  <img
                    src={img1}
                    alt="Image 1"
                    className=" w-20 h-23"
                    style={style || {}}
                  />
                </div>
              </div>
              <div className="justify-center flex flex-col p-3 text-[22px]">
                <h1 className="ml-6 font-extrabold text-xl">{heading2}</h1>
                <h6 className="ml-6 mt-2 leading-6 text-base text-customgrey">
                  {headingpara2}
                </h6>
                <div className="absolute -left-16">
                  <img
                    src={img2}
                    alt="Image 1"
                    className=" w-20 h-23"
                    style={style || {}}
                  />
                </div>
              </div>
              <div className="justify-center flex flex-col p-3 text-[22px]">
                <h1 className="ml-6 font-extrabold text-xl">{heading3}</h1>
                <h6 className="ml-6 mt-2 leading-6 text-base text-customgrey">
                  {headingpara3}
                </h6>
                <div className="absolute -left-16">
                  <img
                    src={img3}
                    alt="Image 1"
                    className=" w-20 h-23"
                    style={style || {}}
                  />
                </div>
              </div>
            </div>
          </div>
          <div id="image_block_01">
            <div
              className="image-box wow slideInRight animate22 animation-banner bannerpic w-full"
              data-wow-delay="0ms"
              data-wow-duration="1500ms"
            >
              <figure className="image js-tilt">
                <img src={serviceBanner} alt="" className="" />
                <div className="js-tilt-glare">
                  <div className="js-tilt-glare-inner"></div>
                </div>
              </figure>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default ChooseUs;
