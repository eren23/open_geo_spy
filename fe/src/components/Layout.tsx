import type { ReactNode } from 'react';

// ---------------------------------------------------------------------------
// Layout -- split-view app shell
//
// Desktop : left panel 60%, right panel (map) 40%, side-by-side
// Mobile  : stacks vertically, left panel on top, map below
// ---------------------------------------------------------------------------

interface LayoutProps {
  /** Left panel content (sidebar / chat / upload) */
  left: ReactNode;
  /** Right panel content (map) */
  right: ReactNode;
}

export default function Layout({ left, right }: LayoutProps) {
  return (
    <div className="flex flex-col md:flex-row h-screen w-screen overflow-hidden bg-gray-100">
      {/* Left panel */}
      <div className="w-full md:w-[60%] h-[50vh] md:h-full overflow-y-auto border-b md:border-b-0 md:border-r border-gray-200 bg-white">
        {left}
      </div>

      {/* Right panel (map) */}
      <div className="w-full md:w-[40%] h-[50vh] md:h-full relative">
        {right}
      </div>
    </div>
  );
}
