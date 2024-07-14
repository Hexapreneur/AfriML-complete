import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronDown } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { useState } from "react";
import { Loader2 } from "lucide-react";

const TrainingCard = ({
  checked,
  setChecked,
  classes,
  isLoading,
  error,
  trainingHandler,
  success,
}) => {
  //  toggleCheckedHandler = () => {
  //     setChecked((prev) => !prev);
  //   };

  return (
    <>
      <div className="flex justify-center ">
        <Card className="w-[200px] rounded-b-2xl shadow-2xl relative end">
          <CardHeader className="px-4 flex gap-4 border-b border-solid border-blue-500 border-opacity-30">
            <CardTitle className="text-start  ">Training</CardTitle>
            <CardDescription className="w-full h-full mt-2">
              <Button
                className="w-full h-full bg-blue-100 text-customgrey-600 hover:bg-blue-200 gap-1"
                onClick={trainingHandler}
                disabled={isLoading}
              >
                {isLoading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  "Train Model"
                )}
              </Button>
              {error && <div className="text-red-500">{error}</div>}
              {success && <div className="text-green-500 pt-1 ">{success}</div>}
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center py-3 px-4 flex justify-between items-center">
            <CardDescription className="text-customgrey ">
              Advanced
            </CardDescription>
            <Checkbox checked={checked} onCheckedChange={setChecked} />
          </CardContent>
        </Card>
      </div>
    </>
  );
};

export default TrainingCard;
