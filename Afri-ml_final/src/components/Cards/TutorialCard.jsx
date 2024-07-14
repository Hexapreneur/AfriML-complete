import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const TutorialCard = ({ cardData }) => {
  const { title, imgSrc, author, commentscount, date, description } = cardData;

  return (
    <Card className="flex max-w-[370px] py-0 border-0 news-block-one">
      <div className="relative bg-white rounded-[10px] overflow-hidden !shadow-custom opacity-[0.93] inner-box">
        <CardHeader className="p-0 image-holder ">
          <img
            className="relative w-full h-full object-cover"
            alt="Tutorial jpg"
            src={imgSrc}
          />
          <div className="flex align-middle items-center justify-center gap-2 absolute w-[124px] h-[40px] top-[22px] left-[30px] bg-white rounded-[30px]">
            <div className="font-normal text-primary text-[16px] tracking-[0] leading-[16px] whitespace-nowrap">
              <svg
                width="15"
                height="17"
                viewBox="0 0 15 17"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M0.709961 14.6582V6.1582H14.71V14.6582C14.71 15.502 14.0225 16.1582 13.21 16.1582H2.20996C1.36621 16.1582 0.709961 15.502 0.709961 14.6582ZM10.71 8.5332V9.7832C10.71 10.002 10.8662 10.1582 11.085 10.1582H12.335C12.5225 10.1582 12.71 10.002 12.71 9.7832V8.5332C12.71 8.3457 12.5225 8.1582 12.335 8.1582H11.085C10.8662 8.1582 10.71 8.3457 10.71 8.5332ZM10.71 12.5332V13.7832C10.71 14.002 10.8662 14.1582 11.085 14.1582H12.335C12.5225 14.1582 12.71 14.002 12.71 13.7832V12.5332C12.71 12.3457 12.5225 12.1582 12.335 12.1582H11.085C10.8662 12.1582 10.71 12.3457 10.71 12.5332ZM6.70996 8.5332V9.7832C6.70996 10.002 6.86621 10.1582 7.08496 10.1582H8.33496C8.52246 10.1582 8.70996 10.002 8.70996 9.7832V8.5332C8.70996 8.3457 8.52246 8.1582 8.33496 8.1582H7.08496C6.86621 8.1582 6.70996 8.3457 6.70996 8.5332ZM6.70996 12.5332V13.7832C6.70996 14.002 6.86621 14.1582 7.08496 14.1582H8.33496C8.52246 14.1582 8.70996 14.002 8.70996 13.7832V12.5332C8.70996 12.3457 8.52246 12.1582 8.33496 12.1582H7.08496C6.86621 12.1582 6.70996 12.3457 6.70996 12.5332ZM2.70996 8.5332V9.7832C2.70996 10.002 2.86621 10.1582 3.08496 10.1582H4.33496C4.52246 10.1582 4.70996 10.002 4.70996 9.7832V8.5332C4.70996 8.3457 4.52246 8.1582 4.33496 8.1582H3.08496C2.86621 8.1582 2.70996 8.3457 2.70996 8.5332ZM2.70996 12.5332V13.7832C2.70996 14.002 2.86621 14.1582 3.08496 14.1582H4.33496C4.52246 14.1582 4.70996 14.002 4.70996 13.7832V12.5332C4.70996 12.3457 4.52246 12.1582 4.33496 12.1582H3.08496C2.86621 12.1582 2.70996 12.3457 2.70996 12.5332ZM13.21 2.1582C14.0225 2.1582 14.71 2.8457 14.71 3.6582V5.1582H0.709961V3.6582C0.709961 2.8457 1.36621 2.1582 2.20996 2.1582H3.70996V0.658203C3.70996 0.408203 3.92871 0.158203 4.20996 0.158203H5.20996C5.45996 0.158203 5.70996 0.408203 5.70996 0.658203V2.1582H9.70996V0.658203C9.70996 0.408203 9.92871 0.158203 10.21 0.158203H11.21C11.46 0.158203 11.71 0.408203 11.71 0.658203V2.1582H13.21Z"
                  fill="#6377EE"
                />
              </svg>
            </div>
            <div className="[font-family:'Roboto-Regular',Helvetica] font-normal text-darkgrey text-[17px] tracking-[0] leading-[32px] whitespace-nowrap">
              {date}
            </div>
          </div>
        </CardHeader>
        <CardContent className="relative pl-7 pt-6 flex flex-col items-start justify-center">
          <div className="flex align-middle items-center justify-start gap-3 py-0 ">
            <div className="[font-family:'Roboto-Regular',Helvetica] text-customgrey text-[14px] md:text-[15px] lg:text-[17px] tracking-[0] leading-[32px] whitespace-nowrap">
              by {author}
            </div>
            <div className=" w-px h-[12px] bg-customgrey" />
            <div className="[font-family:'Roboto-Regular',Helvetica] text-customgrey text-[14px] md:text-[15px] lg:text-[17px] tracking-[0] leading-[32px] whitespace-nowrap">
              {commentscount} Comments
            </div>
          </div>
          <div className="inline-flex pb-1">
            <CardTitle className="relative text-start font-semibold text-grey text-[16px] md:text-[20px] lg:text-[22px] tracking-[0] leading-[32px]">
              {title}
            </CardTitle>
          </div>
          <div className="flex ">
            <CardDescription className="relative text-start [font-family:'Roboto-Regular',Helvetica] text-customgrey text-[14px] md:text-[15px] lg:text-[17px] tracking-[0] leading-[32px]">
            {description}
            </CardDescription>
          </div>
        </CardContent>
      </div>
    </Card>
  );
};

export default TutorialCard;
