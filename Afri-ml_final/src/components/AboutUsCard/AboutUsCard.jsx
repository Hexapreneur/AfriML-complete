import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
const AboutUsCard = ({ title, description, children }) => {

  return (
    <>
      <Card className="w-[430px] h-64 border-0 !shadow-custom rounded-2xl">
        <CardHeader className="gap-4">
          <CardTitle className="text-start  ">
            {title}
          </CardTitle>
          <CardDescription className="text-start text-xs leading-loose text-customgrey">
            {description}
          </CardDescription>
          {children}
        </CardHeader>
      </Card>
    </>
  );
};

export default AboutUsCard;
