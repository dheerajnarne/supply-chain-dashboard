import React, { createContext, useState, useContext, useEffect } from 'react';
import toast from 'react-hot-toast';
import apiService from '../api/api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const initAuth = async () => {
            const token = localStorage.getItem('token');
            if (token) {
                try {
                    const response = await apiService.getCurrentUser();
                    setUser(response.data);
                } catch (error) {
                    console.error('Auth initialization failed', error);
                    localStorage.removeItem('token');
                }
            }
            setLoading(false);
        };

        initAuth();
    }, []);

    const login = async (email, password) => {
        try {
            const response = await apiService.login(email, password);
            const { access_token } = response.data;
            localStorage.setItem('token', access_token);
            const userResponse = await apiService.getCurrentUser();
            setUser(userResponse.data);
            toast.success(`Welcome back, ${userResponse.data.full_name || 'User'}!`);
            return userResponse.data;
        } catch (error) {
            toast.error(error.response?.data?.detail || 'Login failed');
            throw error;
        }
    };

    const signup = async (email, password, fullName) => {
        try {
            await apiService.signup(email, password, fullName);
            toast.success('Account created successfully!');
            // Auto login after signup
            return login(email, password);
        } catch (error) {
            toast.error(error.response?.data?.detail || 'Signup failed');
            throw error;
        }
    };

    const logout = () => {
        localStorage.removeItem('token');
        setUser(null);
        toast.success('Logged out successfully');
    };

    return (
        <AuthContext.Provider value={{ user, login, signup, logout, loading }}>
            {!loading && children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
