const Model = ({ modelData }) => {
  const { title, imgSrc, description, imgSrc1 } = modelData;

  return (
    <div className="relative w-[370px] pr-20 flex flex-col items-start justify-start gap-4">
      <img src={imgSrc1} alt="" className="relative" />
      {imgSrc}
      <div className="pl-2 font-semibold text-darkgrey text-[16px] md:text-[20px] lg:text-[22px] tracking-[0] leading-[30px] whitespace-nowrap">
        {title}
      </div>
      <p className="pl-2 text-start text-grey text-[14px] md:text-[15px] lg:text-[17px] tracking-[0] leading-[32px]">
        {description}
      </p>
    </div>
  );
};

export default Model;
