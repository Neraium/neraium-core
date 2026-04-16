"use client";

import { ReactNode } from "react";
import { Sidebar } from "./Sidebar";

type Props = {
  children: ReactNode;
};

export function AppShell({ children }: Props) {
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main">
        {children}
      </div>
    </div>
  );
}
