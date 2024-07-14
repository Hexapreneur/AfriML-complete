import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import banner1 from "@/assets/banner-1.png";
import banner2 from "@/assets/banner-2.png";
import banner3 from "@/assets/banner-5.png";
import banner4 from "@/assets/banner-3.png";
import banner5 from "@/assets/banner-4.png";

const Banner = () => {
  return (
    <>
      <div className="bannermain flex w-3/4 mr-auto ml-auto mb-20 mt-36 ">
        <div className="bannerheading w-10/12 z-10 items-center flex ">
          <div className="content-box flex flex-col items-start ">
            <h1 className="sm:text-4xl md:text-5xl text-2xl leading-tight font-bold text-left ">
              Machine
              <br /> Learning for <br />
              <span className="text-primary">African </span> Kids.
            </h1>
            <br />
            <h3 className=" text-left font-semibold">
              Train a computer to recognize your <br />
              own images, sounds, text & poses.
            </h3>
            <br />
            <Button
              variant="secondary"
              className="shadow-xl rounded-full text-white px-8"
              asChild
            >
              <Link to="/models" className="z-10">
                Get Started
              </Link>
            </Button>
          </div>
        </div>
        <div
          className="image-box wow slideInRight animate22 animation-banner bannerpic w-10/12"
          data-wow-delay="0ms"
          data-wow-duration="1500ms"
        >
          <figure className="image image-1">
            <img src={banner1} alt="" />
          </figure>
          <figure className="image image-2 float-bob-y">
            <img src={banner2} alt="" />
          </figure>
          <figure
            className="image image-3 wow slideInDown animate22 animation-banner "
            data-wow-delay="500ms"
            data-wow-duration="1500ms"
          >
            <img src={banner3} alt="" />
          </figure>
          <figure className="image image-4">
            <img src={banner4} alt="" />
          </figure>
          <figure className="image image-5 rotate-me">
            <img src={banner5} alt="" />
          </figure>
        </div>
      </div>
    </>
  );
};

export default Banner;
