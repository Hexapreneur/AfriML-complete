import { Button } from "@/components/ui/button";
import logo from "@/assets/logo.png";
import { Link } from "react-router-dom";
import { CircleUserRound } from "lucide-react";

const Navbar = () => {
  return (
    <>
      <div className="main flex justify-between pt-10">
        <div>
          <Link to="/">
            <img src={logo} alt="Logo" />
          </Link>
        </div>
        <div className="flex items-center ">
          <Button
            className="flex text-darkgrey uppercase font-semibold text-base"
            variant="link"
            asChild
          >
            <Link to="/image" className="z-10">
              Image Model
            </Link>
          </Button>
          <Button
            className="flex text-darkgrey uppercase font-semibold text-base"
            variant="link"
            asChild
          >
            <Link to="/models" className="z-10">
              Models
            </Link>
          </Button>
          <Button
            className="flex text-darkgrey uppercase font-semibold text-base"
            variant="link"
            asChild
          >
            <Link to="/about" className="z-10">
              About
            </Link>
          </Button>
          <Button
            className="flex text-darkgrey uppercase font-semibold text-base"
            variant="link"
          >
            <Link to={"/faq"}>Faq</Link>
          </Button>
          {/* <Button className="!shadow-custom bg-background hover:bg-popover text-black rounded-full px-8 ml-10 font-semibold text-base">
            Login
          </Button> */}
          <CircleUserRound className="ml-4" size={32} />
        </div>
      </div>
    </>
  );
};

export default Navbar;
