import { useState, useEffect, useRef, useCallback } from "react";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import { TestimonialCard } from "@/components/Cards/TestimonialCard";
import Autoplay from "embla-carousel-autoplay";

const DotButton = (props) => {
  const { children, ...restProps } = props;

  return (
    <button type="button" {...restProps}>
      {children}
    </button>
  );
};

export function TestimonialCarousel() {
  const [api, setApi] = useState();
  const [current, setCurrent] = useState(0);
  const [count, setCount] = useState([]);

  const plugin = useRef(Autoplay({ delay: 3000, stopOnInteraction: false }));
  const scrollTo = useCallback((index) => api && api.scrollTo(index), [api]);

  useEffect(() => {
    if (!api) {
      return;
    }

    setCount(api.scrollSnapList());
    setCurrent(api.selectedScrollSnap());

    api.on("select", () => {
      setCurrent(api.selectedScrollSnap());
    });
  }, [api]);

  return (
    <Carousel
      className="w-full max-w-xl"
      opts={{
        align: "start",
        loop: true,
      }}
      plugins={[plugin.current]}
      setApi={setApi}
    >
      <CarouselContent>
        {Array.from({ length: 5 }).map((_, index) => (
          <CarouselItem key={index} className="px-4 mt-20">
            <div className="p-1">
              <TestimonialCard />
            </div>
          </CarouselItem>
        ))}
      </CarouselContent>
      <div className="embla__dots">
        {count.map((_, index) => (
          <DotButton
            key={index}
            onClick={() => scrollTo(index)}
            className={"w-1 h-1 rounded-full bg-primary mx-2".concat(
              index === current
                ? " ring-offset-[3px] ring-1 ring-primary !rounded-full"
                : ""
            )}
          />
        ))}
      </div>
    </Carousel>
  );
}
