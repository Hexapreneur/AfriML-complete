import React from "react";
import testimonialbg from "@/assets/testimonialbg.png";
import { TestimonialCarousel } from "@/components/Carousel/carousel";

const Testimonial = () => {
  return (
    <div className="flex flex-col w-full justify-center items-center relative py-20">
      <img
        className="absolute top-0 left-0 w-full h-full object-cover"
        alt="Group"
        src={testimonialbg}
      />
      <div className="flex flex-col w-full items-center gap-20 relative ">
        <div className="flex flex-col w-full items-start gap-4 relative ">
          <div className="flex flex-col w-full items-center gap-2 relative ">
            <div className="relative w-fit font-medium text-primary text-4 text-center tracking-[3.00px] whitespace-nowrap">
              TESTIMONIALS
            </div>
            <div className="relative w-fit font-bold text-darkgrey text-4xl text-center leading-tight whitespace-nowrap">
              What Users Say?
            </div>
          </div>
          <div className="w-full relative flex flex-col items-center">
            <p className="relative w-fit font-normal text-customgrey text-lg text-center">
            Afri-ML exceeded my expectations! The program was enlightening and enjoyable. <br />
            I gained a wealth of knowledge on ML concepts that surpassed my prior <br />
            understanding, and I am grateful to have used it.
            </p>
          </div>
        </div>
        <TestimonialCarousel />
      </div>
    </div>
  );
};

export default Testimonial;
