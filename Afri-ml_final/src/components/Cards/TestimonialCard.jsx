import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import avatar from "@/assets/testimonial.png";

export function TestimonialCard() {
  return (
    <Card className="relative !shadow-custom border-0 mb-3 ml-3 bg-white">
      <CardHeader className="testimonial-png-wrapper flex justify-center pb-12">
        <Avatar className="w-24 h-24 absolute -top-14 left-1/2 -translate-x-1/2">
          <AvatarImage src={avatar} />
          <AvatarFallback>CN</AvatarFallback>
        </Avatar>
      </CardHeader>
      <CardContent className="px-14">
        <CardDescription className="text-center text-customgrey">
          &#34;Just had an incredible experience using Afri-ML! 🚀 The
          platform&#39;s user-friendly interface made it easy to dive into
          machine learning. Kudos to the team for creating such an engaging
          tool!&#34;
        </CardDescription>
      </CardContent>
      <CardFooter className="flex justify-center items-center flex-col gap-2 pb-10">
        <CardTitle className="text-medium text-lg whitespace-nowrap ">
          Nicolas Lawson
        </CardTitle>
        <div className="text-wrapper text-primary ">Designer</div>
      </CardFooter>
    </Card>
  );
}
