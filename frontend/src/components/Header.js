// src/components/Header.js
import React from 'react';
import './Header.css';
import logo from '../assets/logo.svg'; // You'll need to create this

const Header = () => {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <div className="logo">
            <img src={logo} alt="CompileML Logo" className="logo-img" />
            <span className="logo-text">CompileML</span>
          </div>
          <nav className="nav">
            <ul className="nav-list">
              <li className="nav-item">
                <a href="/" className="nav-link active">Dashboard</a>
              </li>
              <li className="nav-item">
                <a href="/docs" className="nav-link">Documentation</a>
              </li>
              <li className="nav-item">
                <a href="/settings" className="nav-link">Settings</a>
              </li>
            </ul>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;