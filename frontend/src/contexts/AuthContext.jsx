import { createContext, useContext, useState, useEffect } from 'react'
import {
    onAuthStateChanged,
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    signInWithPopup,
    GoogleAuthProvider,
    signOut,
    updateProfile,
    sendEmailVerification
} from 'firebase/auth'
import { auth } from '../firebase'

const AuthContext = createContext(null)

export function useAuth() {
    const ctx = useContext(AuthContext)
    if (!ctx) throw new Error('useAuth must be used within an AuthProvider')
    return ctx
}

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
            setUser(firebaseUser)
            setLoading(false)
        })
        return unsubscribe
    }, [])

    const login = async (email, password) => {
        const cred = await signInWithEmailAndPassword(auth, email, password)
        if (!cred.user.emailVerified) {
            await signOut(auth)
            const err = new Error('Please verify your email address to log in.')
            err.code = 'auth/unverified-email'
            throw err
        }
        return cred
    }

    const signup = async (email, password, displayName) => {
        const cred = await createUserWithEmailAndPassword(auth, email, password)
        if (displayName) {
            await updateProfile(cred.user, { displayName })
        }
        await sendEmailVerification(cred.user)
        await signOut(auth)

        const err = new Error('Signup successful! Please check your email to verify your account before logging in.')
        err.code = 'auth/signup-success-unverified'
        throw err
    }

    const loginWithGoogle = () =>
        signInWithPopup(auth, new GoogleAuthProvider())

    const logout = () => signOut(auth)

    const getToken = async () => {
        if (!auth.currentUser) return null
        return auth.currentUser.getIdToken()
    }

    const value = { user, loading, login, signup, loginWithGoogle, logout, getToken }

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
