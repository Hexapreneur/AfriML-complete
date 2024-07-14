import { Card, CardFooter, CardHeader } from "@/components/ui/card";

function ProjectCard({ cardData }) {
  const { title, imgSrc, description } = cardData;

  return (
    <Card className="flex max-w-[370px] relative bg-white project-block-one">
      <div className="w-full relative rounded-xl !shadow-custom">
        <CardHeader className="p-0 image-holder">
          <img
            className="relative w-full h-full object-cover"
            alt="Project jpg"
            src={imgSrc}
          />
        </CardHeader>
        <CardFooter className="flex flex-col !items-start w-[370px] pl-[40px] pr-[165px] pt-[24px] pb-[23px] relative ">
          <div className="relative font-semibold text-darkgrey text-[16px] md:text-[20px] lg:text-[22px] tracking-[0] leading-[30px] whitespace-nowrap">
            {title}
          </div>
          <div className="relative [font-family:'Roboto-Regular',Helvetica] font-normal text-customgrey text-[14px] md:text-[15px] lg:text-[17px] tracking-[0] leading-[28px] whitespace-nowrap">
            {description}
          </div>
        </CardFooter>
      </div>
    </Card>
  );
}

export default ProjectCard;
