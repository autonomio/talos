import React from 'react';
import {useThemeConfig} from '@docusaurus/theme-common';
import NavbarItem from '@theme/NavbarItem';
import NavbarLogo from '@theme/Navbar/Logo';
import NavbarColorModeToggle from '@theme/Navbar/ColorModeToggle';
import SearchBar from '@theme/SearchBar';

export default function Navbar() {
  const {navbar} = useThemeConfig();
  return (
    <nav className="navbar autonomio-navbar" aria-label="Main">
      <div className="navbar__inner autonomio-wrapper">
        <NavbarLogo />
        <div className="navbar__items autonomio-navigation">
          {navbar.items.map((item, index) => <NavbarItem key={index} {...item} />)}
        </div>
        <div className="autonomio-tools">
          <SearchBar />
          <NavbarColorModeToggle />
        </div>
      </div>
    </nav>
  );
}
