import workingbannerpattern from "../../assets/workingbannerpattern.png";
import workingbannerpic from "../../assets/workingbannerpic.png";
import workingbannervector from "../../assets/workingBannervector.png";
import bg2 from "@/assets/rsz_logo_copy.png";

const WorkingBanner = () => {
  return (
    <>
      <div className="woorkingmain  w-full   ml-auto mr-auto bg-[#F7FAFF] pb-28 pt-10">
        <img src={bg2} alt="" className=" absolute w-1/4 right-0" />
        <div className="absolute  w-1/4 right-0 z-0 w-[400px] h-[400px] bg-[#F7FAFF]">
          <img
            className="z-1 absolute right-0 w-[200px] h-[200px] opacity-10"
            alt="Group"
            src={bg2}
          />
        </div>
        <div className="woorkingmain  ml-auto mr-auto   w-3/4 h-full flex z-50 items-center">
          <div className="working w-1/2 h-3/5 text-left">
            <span className="text-md text-primary">HOW IT WORKS</span>
            <br />
            <br />
            <h2 className="sm:text-4xl md:text-3xl text-2xl leading-tight font-extrabold text-left ">
              How Afri-ML Works?
            </h2>
            <br />
            <span className="text-sm text-customgrey">
              It works like a fun game where African kids can teach the
              computer to recognise images, audio, text and poses. It can also 
              detect African objects, accent and language.
            </span>
            <br />
            <br />
            <div className="flex flex-col relative">
              <div className="circle text-primary w-14 bg-white !shadow-custom justify-center flex rounded-full p-3 text-[22px] ">
                1{" "}
              </div>
              <img
                src={workingbannervector}
                alt=""
                className="w-[14px] h-20 ml-[21px]  text-sm"
              />
              <div className="circle text-primary w-14 bg-white !shadow-custom justify-center flex rounded-full p-3 text-[22px] ">
                2{" "}
              </div>
              <img
                src={workingbannervector}
                alt=""
                className="w-[14px] h-20 ml-[21px] "
              />
              <div className="circle text-primary w-14 bg-white !shadow-custom justify-center flex rounded-full p-3 text-[22px] ">
                3{" "}
              </div>
              <img
                src={workingbannervector}
                alt=""
                className="w-[14px] h-20 ml-[21px] "
              />
              <div className="circle text-primary w-14 bg-white !shadow-custom justify-center flex rounded-full p-3 text-[22px] ">
                4{" "}
              </div>
              <div className="text top-0 absolute left-14 ">
                <h1 className="ml-6 font-extrabold">Gather Data</h1>
                <h6 className="ml-6 text-xs mt-2 leading-6">
                  Gather and group your examples into classes, or <br />
                  categories, that you want the computer to learn.
                </h6>
                <br />
                <br />
                <h1 className="ml-6 font-extrabold">Train Model</h1>
                <h6 className="ml-6 text-xs mt-2 leading-6">
                  Train your model, then instantly test it out to see whether it{" "}
                  <br />
                  can correctly classify new examples.
                </h6>
                <br />
                <br />
                <h1 className="ml-6 font-extrabold">Test & Detect</h1>
                <h6 className="ml-6 text-xs mt-2 leading-6">
                  Test your model for accuracy and detect any African cultural
                  <br />
                  element using our provided detecting agents.
                </h6>
                <br />
                <br />
                <h1 className="ml-6 font-extrabold">Export</h1>
                <h6 className="ml-6 text-xs mt-2 leading-6">
                  Export your model for your projects: sites, apps, and more.
                  <br /> You can download your model or host it online.
                </h6>
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
                <img src={workingbannerpic} alt="" />
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

export default WorkingBanner;
