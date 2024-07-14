import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
// import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";

export function Innercard({ cardData }) {
  const { title, imgSrc1, imgSrc2, imgSrc3, description, buttonText, link } =
    cardData;
  return (
    <Card className="sm:w-4/5 md:w-1/2 lg:w-3/12 relative !shadow-custom border-0 pb-10  ">
      <div className=" flex justify-center items-center relative ">
        <img src={imgSrc2} alt="" className="absolute" />
        <img src={imgSrc1} alt="" />
        {imgSrc3}
      </div>
      <CardHeader>
        <CardTitle className="text-medium text-xl whitespace-nowrap  ">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="px-14">
        <CardDescription className=" text-customgrey ">
          <span className="  h-10 text-sm  flex justify-center">
            {description}
          </span>
        </CardDescription>
      </CardContent>
      <CardFooter className="flex justify-center items-center flex-col gap-2 ">
        <Button
          variant="outline"
          className="rounded-full text-primary border-primary px-6 mt-10"
          asChild
        >
          <Link to={link} className="z-10">
            {buttonText}
          </Link>
        </Button>
      </CardFooter>
    </Card>
  );
}

export default Innercard;
